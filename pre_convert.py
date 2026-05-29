import datetime
import re
import json
import aiocache
import copy
import asyncio
import random
import time
import numpy as np
import pandas as pd
import jionlp as jio
from typing import Any, Optional, Dict, List
from collections import defaultdict
from database.dataloader import DataLoader
from state import PipelineState
from shared.util import Util, StringUtil
from fuzzy.fuzzy_match import FuzzyMatch

import logging
logger = logging.getLogger(__name__)
import itertools

# 前置框架
class PreConverter:
    INDEX_DIM_DEFAULTS_DICT = {}
    index_split: pd.DataFrame = pd.DataFrame()
    day_values: List[str] = []
    dim_map: Dict[str, str] = {}
    table_model_dict: Dict[str, Any] = {}
    middle_index_name_code_dict: Dict[str, str] = {}

    @classmethod
    def init(cls):
        # 初始化 INDEX_DIM_DEFAULTS_DICT
        cls.INDEX_DIM_DEFAULTS_DICT = {
            "统计口径": ["当月", "当日", "月累计", "年累计"],
            "投诉流程": ["整体投诉"],
            "投保环节": ["承保"],
            "指标保单类型": [None],
            "渠道": ["监管全量"],
            "客户类型": ["全量"],
            "主题": ["主题"],
        }

        # 初始化 index_split DataFrame
        cls.index_split = pd.DataFrame([
            {
                "元子指标名称": "监管投诉引导率",
                "意图": "个险_投诉日报_按受理时间",
                "统计口径": "当月",
                "频率": "月",
                "捷报名称": "监管投诉引导率",
                "中台中文名称": "监管投诉引导率",
                "绑定维度(可忽略)": "投诉流程;投保环节;指标保单类型",
                "中台指标编码": "INDEX_CODE_1",
                "投诉流程": "整体投诉",
                "投保环节": "承保",
                "指标保单类型": None,
            },
        ])

        # 初始化 day_values 列表
        cls.day_values = ["日", "号", "天"]

        # 初始化 dim_map 字典
        cls.dim_map = {
            "个险_投诉日报_按受理时间监管投诉引导率": "投诉流程;投保环节;指标保单类型",
        }

        # 初始化 table_model_dict 字典
        cls.table_model_dict = {
            "个险_投诉日报_按受理时间意图": None,
        }

        # 初始化 middle_index_name_code_dict 字典
        cls.middle_index_name_code_dict = {
            "个险_投诉日报_按受理时间监管投诉引导率": "INDEX_CODE_1",
        }

    @classmethod
    def _find_frequency_keyword(cls, user_question, default_freq=None, standard_index_name=None):
        if re.search(r'(?:日|号|当日)', user_question):
            return '日'
        if re.search(r'(?:月|当月|本月|下月|下下月)', user_question):
            return '月'
        if re.search(r'(?:年|今年|明年)', user_question):
            return '年'
        if re.search(r'(?:季|季度)', user_question):
            return '季'
        return default_freq or '月'

    @classmethod
    async def _prior_elements(cls, lst):
        if not lst:
            return lst
        priority = ['通用', '绩优', '产品']
        ordered = [item for item in priority if item in lst]
        ordered.extend([item for item in lst if item not in ordered])
        return ordered

    @classmethod
    async def execute(cls, state: PipelineState) -> PipelineState:
        valid_table_scopes = DataLoader.table_names
        # step1--指标抽取
        index_time = time.time()
        state.original_question = state.question
        state.question = await cls.question_rewrite(state.question)
        state = await cls.index_convert(
            state=state, valid_table_scopes=valid_table_scopes
        )
        logger.info(f"[step1.1--指标抽取]{state.request_id}]{state.unconfirmed_dimensions}{state.unconfirmed_metrics}")
        logger.info(f"[step1.1--指标抽取耗时]{state.request_id}]{time.time() - index_time}")

        # step2--属性识别
        # Note: 目前没有迁移属性识别
        attr_results = []
        # step2.3--主题指标填充
        attr_results, topic_filling = await cls.topic_recognize(state=state, attr_results=attr_results)
        logger.info(f"[step2.3--主题指标填充]{state.request_id}]{attr_results}")
        PREDICT_INDEX = False
        if PREDICT_INDEX:
            attr_results = cls.index_predict(attr_results=attr_results)
        # step2.4--指标还原&维度删除
        attr_results, question_info_list, ret_record = await cls.index_resolve(
            state, attr_results, valid_table_scopes, topic_filling
        )
        logger.info(f"[step2.4--指标还原&维度删除]{state.request_id}]{attr_results}")

        return state

    # HIGH PRIORITY 指标-1
    @classmethod
    async def question_rewrite(cls, question: str) -> str:
        """问题改写

        background:
            1.需要识别业务领域的黑话
            2.关于'新人'的指标识别的转换

        solution:    
            - 短期方案：直接将相关代码挪到前置框架
                1. rewrite from index_extraction.py: add_words
                2. add convert for '新人' (learn from index_extraction.py: execute), '直达转办'还是放在index_resolve
            - 长期方案：evidence；关键词展开；相关代码@颍楠

        Args:
            question(str): 原始问题

        Returns:
            str: 改写后的问题
        """
        if not question:
            return question

        # 规则：寿钻-纯寿钻石人力，寿钻会-纯寿钻石会员
        if "寿钻" in question and "纯寿钻石" not in question:
            if "寿钻会" in question and "寿钻会员" not in question:
                question = question.replace("寿钻会", "纯寿钻石会员")
            elif "寿钻" in question and "寿钻会员" not in question:
                question = question.replace("寿钻", "纯寿钻石人力")
            else:
                question = question.replace("寿钻", "纯寿钻石")

        # 规则：地理位置简称展开
        question = question.replace("北上广深", "北京、上海、广东、深圳")
        question = question.replace("北上广", "北京、上海、广东")
        question = question.replace("豫南", "河南南部")
        question = question.replace("豫北", "河南北部")

        # 规则：M+Q，同环比
        question = question.replace("M+Q主体", "MVP和QVP")
        question = question.replace("M+Q", "MVP和QVP")
        question = question.replace("同环比", "同比和环比")
        question = question.replace("环同比", "环比和同比")

        question = question.replace("top", "排名前")
        question = question.replace("TOP", "排名前")

        question = question.replace("新活", "新人有效活动率")
        question = question.replace("销售总", "销售总监")
        question = question.replace("短赔", "短险赔付率")
        question = question.replace("主管数量", "主管人数")
        question = question.replace("各钻石人力", "各层级钻石人力")
        question = question.replace("试用业务员", "试用职级的在职人力")
        question = question.replace("三好五星占比", "三好五星星级部课占比")
        question = question.replace("N达成", "NBEV达成")
        question = question.replace("各钻石会员", "各钻石层级")
        
        question = question.replace("哪一年", "哪年")
        
        # 规则：开门红
        if "开门红" in question:
            ner_time_result = jio.ner.extract_time(question, time_base=time.time())
            if not ner_time_result:
                current_month = datetime.datetime.now().month
                new_word = "今年" if current_month <= 9 else "明年1-3月"
                question = question.replace("开门红", new_word)
            else:
                question = question.replace("开门红", "1-3月")
        
        # 规则：监管通报件
        ## 1219: 内部投诉件15日结案率, 内部投诉件24小时响应率
        if "结案率" not in question and "响应率" not in question:
            question = re.sub(
                r'(监管全量|监管引导|内部投诉|内诉|整体投诉|监管通报|投诉通报|监管(首次|二次|三次及以上|三次以上)?转办)(件数?|量)',
                r'\1案件量',
                question,
            )
        
        # 规则：主任晋升培训普及率
        if "普及" in question:
            question = question.replace("主任晋升培训", "晋升培训").replace("主任晋升", "晋升培训")
        
        # 规则：排名第几替换为排在第几
        question = question.replace("排名第几", "排在第几")

        #标杆：TOP1，同类标杆TOP3则不需要替换
        if re.search('标杆', question) and (not re.search('标杆(前|top)[1-9一二三四五六七八九十]', question)):
            if re.search('系统标杆', question):
                question = question.replace('标杆', '排第1')
            else:
                question = question.replace('标杆', '排名第1')

        if re.search('转办件', question) and (not re.search('(保全)转办件', question)):
            question = question.replace('转办件', '监管转办案件量')

        if re.search('通报件', question):
            question = question.replace('通报件', '通报案件量')

        if re.search('投诉量', question):
            question = question.replace('投诉量', '投诉案件量')
        
        ##年进度 1031版本
        if re.search('年进度', question):
            question = question.replace('年进度', '今年截止目前')

        ##分公司
        question = question.replace("辽分", "辽宁分公司")
        
        ## 12378
        question = question.replace("12378投诉", "年除重-监管引导客户量")

        ## NOTE: 以下原本为非direct执行的rewrite规则
        # 规则：X钻会-X钻钻石会员，X钻人力-X钻钻石人力
        question = re.sub(r'(银|金|双金|单)钻会', r'\1钻钻石会员', question)
        question = re.sub(r'(银|金|双金|单)钻人力', r'\1钻钻石人力', question)
        question = question.replace("新客", "新增投保客户数")

        if re.search('钻石(分析|情况)$', question):
            question = re.sub('钻石', '钻石人力', question)

        # 规则：新人
        # if "普及" not in question and "主管陪访率" not in question and (not re.search('代理人|钻石', question)):
            # question = question.replace("新人", "司龄一年以内的在职人力")
        question = re.sub(r'新人(?:数量|量|数)?', 
                            lambda m: "一年内入职的在职人力数量" if ("数" in m.group(0) or "量" in m.group(0)) else 
                            "一年内入职的在职人力", question)

        return question
    
    # HIGH PRIORITY 指标-2
    @classmethod
    async def index_convert(cls, state: PipelineState, valid_table_scopes: List) -> PipelineState:
        """指标识别的转换

        background:
            比如用户问"直达转办",其对应的指标客户数，维度=直达转办， 指标=客户数。需要对指标进行转换。

        solution:    
            - '新人' 挪到 question_rewrite（add_words）中
                rewrite from index_extraction.py: execute
            - '预收'、'承保'、'纯寿'、'全口径' 放入到指标纬度识别后，加入“指标清洗”功能
                测试后决定是否添加

        Args:
            state(PipelineState): PipelineState对象，包含问题和其他相关信息
            valid_table_scopes(List): 可用表范围
            throw_exception(bool): 是否抛出异常，False时返回{}

        Returns:
            Dict: 识别结果
        """
        # 规则：直达转办相关
        if re.search(r'(直达..客户|客户..直达)', state.question):
            state.unconfirmed_metrics = "客户数"
            state.unconfirmed_dimensions = ["直达转办"]
            state.question = re.sub(r'(直达.*?客户|客户.*?直达)', '客户', state.question)
        
        # 规则：预收、承保、纯寿、全口径等词的处理
        # TODO: 可以考虑在指标纬度识别后加入“指标清洗”功能，针对这些词进行特殊处理

        return state

    @classmethod
    async def _find_statcaliber(cls,user_question,attr_results):
        ner_time_result = jio.ner.extract_time(user_question, time_base=time.time())
        touch_time_result = re.findall(r'(?:日|号|天|月|季|年)', user_question)
        ner_time_result = ner_time_result or touch_time_result

        # 提取元子指标
        index_names = []
        index_stati_dict = {}
        stati_words = []
        all_table_names = []
        for attr_result in attr_results:
            index_result = Util.read_map_value(attr_result, "indexResult", {})
            attr_vals = Util.read_map_value(attr_result, "attrVal", [])
            stati_words += [cur_item['word'] for item in attr_vals for cur_item in item['word_list'] if cur_item['columns_name'] == '统计口径']
            for question_name, index_value in index_result.items():
                for entry in index_value:
                    index_names.extend(entry["indexName"])
                    all_table_names.append(entry["tableName"])
        index_names = list(set(index_names))
        stati_words = list(set(stati_words))
        new_index_vals = {"统计口径": stati_words}
        all_table_names = list(set(all_table_names))

        for index_name in index_names:
            df_index = cls.index_split[cls.index_split["元子指标名称"] == index_name].copy()
            df_index = df_index[df_index["意图"].isin(all_table_names)]
            candidates_original = list(set(df_index["统计口径"]))
            # 删除空值
            candidates_original = [value if value == value else ""
                                   for value in candidates_original]

            new_question = copy.deepcopy(user_question)

            temp_attrs = []
            temp_index_tables = []
            for items in attr_results:
                temp_tables = [value['tableName'] for key,values in items['indexResult'].items() for value in values if value['indexName'] == index_name]
                temp_tables = list(set(temp_tables))
                if temp_tables:
                    temp_attrs.append(items)
                temp_index_tables.extend(temp_tables)

            if ner_time_result:  # 如果句子中有统计口径的词
                # 只保留当月作为默认值，如果候选只有一个，就只用当下一个作为候选值
                if len(candidates_original) == 1:
                    candidates = candidates_original.copy()
                elif len(temp_attrs) == 1:
                    candidates = cls.index_split[
                        (cls.index_split["元子指标名称"] == index_name) &
                        (cls.index_split["意图"].isin(temp_index_tables))
                    ]["统计口径"].tolist()
                    # 删除空值
                    candidates = [value for value in candidates if (value == value and value != "")]
                else:
                    candidates = [value for value in candidates_original if value in ["当月"]]

                # 临时解决方案，需要训练嵌入模型
                for value in cls.day_values:
                    user_question = user_question.replace(value, "")
                if "累计" in user_question or re.findall(r'(?:日|号|天)', user_question):
                    added = False
                    if "累计" in user_question:
                        if "月" in user_question and "年" in user_question and "月累计" in candidates_original:
                            candidates += ["月累计"]
                            added = True
                        elif "月" not in user_question and "年" in user_question and "年累计" in candidates_original:
                            candidates += ["年累计"]
                            added = True
                        elif "月" in user_question and "月累计" in candidates_original:
                            candidates += ["月累计"]
                            added = True
                        elif re.findall(r'(?:日|号|天)', user_question) and "月累计" in candidates_original:
                            candidates += ["月累计"]
                            added = True
                        elif re.findall(r'(?:日|号|天)', user_question) and "年累计" in candidates_original:
                            candidates += ["年累计"]
                            added = True

                    for stat in candidates_original:
                        if "累计" in stat and stat in user_question:
                            candidates += [stat]
                            added = True
                    if re.findall(r'(?:日|号|天)', user_question) and "当日" in candidates_original:
                        candidates += ["当日"]
                        added = True
                        new_question = re.sub(r'([本月|当月|下月|下下月])', '', new_question)

                    if re.findall(r'(?:宽一|应交月上月|下月累计|下月)', user_question) and \
                            "下月" in candidates_original:
                        candidates += ["下月"]
                        added = True

                    if re.findall(r'(?:应交月当月|下下月累计|下下月)', user_question) and \
                            "下下月" in candidates_original:
                        candidates += ["下下月"]
                        added = True

                    if not added:
                        candidates = copy.deepcopy(candidates_original)
                else:
                    if "统计口径" in new_index_vals:
                        candidates += new_index_vals["统计口径"]

            else:  # 句子中没有提到统计口径，默认返回当月？ #1017修改
                all_candidates = candidates_original.copy()
                if len(all_candidates) == 1:
                    candidates = all_candidates
                elif len(temp_attrs) == 1:
                    candidates = cls.index_split[
                        (cls.index_split["元子指标名称"] == index_name) &
                        (cls.index_split["意图"].isin(temp_index_tables))
                    ]["统计口径"].tolist()
                    # 删除空值
                    candidates = [value for value in candidates if (value == value and value != "")]
                elif '当月' in all_candidates:
                    candidates = ['当月']
                else:
                    candidates = candidates_original.copy()

            if candidates:
                candidates = list(set(candidates))
                logger.info(f"指标-{index_name}-的候选统计口径：{candidates}")
                ## 1121新匹配算法
                ## ==========================

                candidates = [cand + question_name for cand in candidates]
                matched_result = await FuzzyMatch.get_embedding_cosine_sim(
                    new_question, candidates
                )
                stat_caliber_embedding = matched_result[0]['word']
                stat_caliber_embedding = stat_caliber_embedding[:-len(question_name)]

                logger.info(f"指标-{index_name}-的最终统计口径：{stat_caliber_embedding}")

                index_stati_dict[index_name] = stat_caliber_embedding

        return index_stati_dict

    # HIGH PRIORITY 指标-3
    @classmethod
    async def index_resolve(cls, state: PipelineState, attr_results: List[Dict], valid_table_scopes: List[str], topic_filter: bool = False) -> PipelineState:
        """基于统计口径等绑定维度筛选指标
        
        background:
            1.使用统计口径和频率来查找正确的指标
            2.某些特定绑定维度需要取默认值

        solution:
            - 短期方案:
                1. rewrite from index_extraction.py: index_locator
                2. 指标还原：纬度识别之后；时间识别之前？？@朋飞确认
            - 长期方案:
                1. 时间识别出来之后，再确定指标口径
                2. 确定逻辑@寿险来做
                3. LLM识别
        
        Args:
            state(PipelineState): PipelineState对象
            attr_results(List[Dict]): 维度识别结果
            valid_table_scopes(List[str]): 可用表范围
            topic_filter(bool): 是否使用业务领域过滤
        """
        ret = []
        question_info_list = []
        other_info_list = []
        data_dict = {}
        valid_table_names = []
        user_question = state.question

        ## 捷报标识:
        jb_flag = False
        if any("捷报" in table_name for table_name in valid_table_scopes):
            jb_flag = True

        ##各个指标的统计口径
        index_stati_dict = await cls._find_statcaliber(state.question, attr_results)

        # 信息提取和处理
        for attr_result in attr_results:
            attr_vals = Util.read_map_value(attr_result, 'attrVal', [])
            attr_objs = Util.read_map_value(attr_result, 'attrObj', [])
            index_result = Util.read_map_value(attr_result, 'indexResult', {})

            # 收集所有元子指标
            index_names = []
            for qname, lst in index_result.items():
                for entry in lst:
                    index_names += entry.get('indexName', [])

            attr_names = []
            for av in attr_vals:
                for we in av.get('word_list', []):
                    attr_names.append(we['columns_name'])

            # 分离指标拆解维度：指标维度(new_index_vals)&原生维度(new_attr_vals)
            new_attr_vals = []
            new_index_vals = {}
            new_index_vals_org = {}
            for av in attr_vals:
                new_word_list = []
                org_word = av.get('org_word')
                for we in av.get('word_list', []):
                    cname = we.get('columns_name')
                    if cname not in cls.INDEX_DIM_DEFAULTS_DICT.keys() and \
                        cname != '主题':
                        # 原生维度保留（暂不修改operator，让答案拼接直接处理）
                        # +++++++++++++++++++++++++++++++++++++++++++++++
                        # 规则：会员与持续对应，人力与产能对应
                        if cname not in ["荣誉体系持续描述", "荣誉体系产能描述"]:
                            new_word_list.append(we)
                        else:
                            if "钻石会员" in index_names or "钻石人力" in index_names:
                                # 钻石会员——荣誉体系持续描述
                                # 钻石人力——荣誉体系产能描述
                                if cname == "荣誉体系持续描述":
                                    if "钻石会员" in index_names:
                                        new_word_list.append(we)
                                if cname == "荣誉体系产能描述":
                                    if "钻石人力" in index_names:
                                        new_word_list.append(we)
                            else:
                                # 非钻石会员/钻石人力，优先返回产能描述
                                if "荣誉体系持续描述" in attr_names and "荣誉体系产能描述" in attr_names:
                                    if cname == "荣誉体系产能描述":
                                        new_word_list.append(we)
                                else:
                                    new_word_list.append(we)
                        # +++++++++++++++++++++++++++++++++++++++++++++++
                    elif (we.get('columns', '').startswith('user_') and cname in ['客户类型']):
                        new_word_list.append(we)
                    elif cname != '主题':
                        if (cname in ["渠道"]) and jb_flag:
                            new_word_list.append(we)

                        # 指标维度分离（用于后续筛选）
                        if cname not in new_index_vals:
                            new_index_vals[cname] = [we.get('word')]
                            new_index_vals_org[cname] = {we.get('word'): org_word}
                        else:
                            new_index_vals[cname].append(we.get('word'))
                            new_index_vals_org[cname][we.get('word')] = org_word

                if new_word_list:
                    av['word_list'] = new_word_list  # 更新word_list
                    new_attr_vals.append(av)  # 更新attrVal

            new_attr_objs = []
            for ao in attr_objs:
                new_word_list = []
                for we in ao.get('word_list', []):
                    if we.get('word') not in cls.INDEX_DIM_DEFAULTS_DICT.keys() and we.get('word') != '主题':
                        # 原生维度保留，指标维度删除
                        new_word_list.append(we)
                    ##0710 当模型维度和用户维度重叠时
                    elif (we.get('columns', '').startswith('user_') and we.get('word') in ['客户类型']):
                        new_word_list.append(we)

                if new_word_list:
                    ao['word_list'] = new_word_list  # 更新word_list
                    new_attr_objs.append(ao)  # 更新attrObj

            # 统计维度乘积
            # {'服务渠道': ['临柜服务'], '客户类型': ['全量', '剔除老年人'], '投诉流程': ['监管全量']}
            index_val_product = 1
            for k, v in new_index_vals.items():
                index_val_product *= len(v)
            index_vals_product = copy.deepcopy(new_index_vals)
            
            
            print("====debug====")
            print("new index vals:", new_index_vals)

            # 指标还原
            # 对于 index_result 中的每个 question_name 和其候选 entry，尝试根据 new_index_vals 进行筛选
            possibilities_index_result = {}
            used_dimensions = []
            for question_name, index_entries in index_result.items():

                #找匹配的维度最多的指标0806
                loc_num = []
                for loc, index_entry in enumerate(index_entries):
                    basic_index_name = index_entry.get('indexName', [])[0] if index_entry.get('indexName') else ""
                    table_name = index_entry.get('tableName')
                    related_columns = cls.dim_map.get(table_name+basic_index_name,"").split(';')
                    value_item = set(related_columns).intersection(list(new_index_vals.keys()))
                    loc_num.append(len(value_item))
                valid_num = [loc for loc,value in enumerate(loc_num) if value == max(loc_num)]

                possibilities_index_entry = []
                for loc, entry in enumerate(index_entries):
                    topic_filling = bool(entry.get("ifTopic"))

                    # 1. 通过元子指标和意图条件筛选指标
                    basic_index_name = entry.get('indexName', [])
                    table_name = entry.get('tableName')
                    index_code = entry.get('indexCode', [])
                    if set(valid_table_scopes) != {"客户"} and not topic_filling:
                        # 非客户域且无指标编码（未经过主题指标填补）
                        df_index = cls.index_split[(cls.index_split["元子指标名称"].isin(basic_index_name)) &
                                                   (cls.index_split["意图"] == table_name)].copy()

                        # 2. 通过频率条件筛选指标（通过关键词找频率，无关键词则输出无）
                        # 筛选后如果候选指标为空，则取消频率条件筛选
                        time_dimension = cls._find_frequency_keyword(state.question)
                        df_index_tmp = df_index[df_index["频率"] == time_dimension].copy()
                        if df_index_tmp.shape[0] >= 1 and not jb_flag:
                            df_index = df_index_tmp.copy()

                        ####1017增加
                        #不同意图相同指标的维度比较
                        index_caliber = [caliber for key,caliber in index_stati_dict.items() if key in basic_index_name]
                        if index_caliber and not jb_flag:
                            df_index_temp = df_index[df_index["统计口径"].isin(index_caliber)].copy()
                            if df_index_temp.shape[0] == 0 and len(attr_results) > 1:
                                logger.info(f"根据别的意图指标统计口径筛选后进行删除")
                                basic_index_names = []
                                news_index_names = []
                                standard_index_names = []
                                continue
                            elif len(attr_results) > 1:
                                df_index = df_index_temp.copy()

                        if df_index.shape[0] == 1:
                            # 3.1 元子指标和意图条件可确认唯一捷报指标
                            basic_index_names = list(df_index["元子指标名称"])
                            news_index_names = list(df_index["捷报名称"])
                            standard_index_names = list(df_index["中台中文名称"])

                        elif df_index.shape[0] > 1:
                            # 3.2 元子指标和意图条件不可确认唯一捷报指标
                            df_filtered = df_index.copy()
                            # 寻找绑定指标
                            related_columns = list(set(df_filtered["绑定维度(可忽略)"]))[0].split(";")

                            # 0806增加
                            if loc not in valid_num:
                                continue

                            print("====debug====")
                            print("new index vals:", new_index_vals)

                            for related_column, related_values in new_index_vals.items():
                                # 只针对统计口径：
                                # 1. 如果一个词是另一个词的子集，就只保留大的字符串
                                # caliber_dict: {'当月': '当月', '下下月': '应交月当月'}
                                if related_column == "统计口径" and len(related_values) > 1:
                                    caliber_dict = new_index_vals_org[related_column]

                                    org_values = [caliber_dict[key] for key in related_values]

                                    new_related_values = []
                                    for val in related_values:
                                        org_val = caliber_dict[val]
                                        other_org_vals = [ov for ov in org_values if ov != org_val]
                                        include_flag = any([org_val in ov for ov in other_org_vals])
                                        if not include_flag:
                                            new_related_values.append(val)
                                    related_values = copy.deepcopy(new_related_values)
                                # 2. 如果存在除了当月和当年之外的值，删除当月和当年统计口径
                                # 当月自保件13月宽一保费继续率
                                if related_column == "统计口径" and len(related_values) > 1:
                                    tmp_values = set(related_values) - {"当月", "当日"}
                                    if tmp_values:
                                        related_values = list(tmp_values)

                                # 3. 只针对投保环节
                                # 当其他投保环节维值和承保同时出现，删除承保
                                if related_column == "投保环节" and len(related_values) > 1 and "承保" in related_values:
                                    related_values = list(set(related_values) - {"承保"})

                                index_vals_product.update({related_column: related_values})

                                # 如果有识别值，则进行筛选
                                # 识别列需要在绑定维度中
                                # 统计口径额外处理
                                ## TODO, 不命中时，按照无值的处理

                                print("=====debug====")
                                print("related columns:", related_columns)
                                print("relate values：", related_values)

                                if related_column in related_columns and \
                                        set(list(df_filtered[related_column])).intersection(set(related_values)) and \
                                        related_column != "主题" and \
                                        (
                                                (related_values and related_column != "统计口径") or
                                                (related_column == "统计口径" and
                                                 {'下下月', '下月', '月累计', '年累计',
                                                  '半年累计', '历史累计', '季累计'}.intersection(set(related_values)))
                                        ) and \
                                        related_column not in used_dimensions:
                                    df_filtered = df_filtered[df_filtered[related_column].isin(related_values)].copy()
                                    related_columns = list(set(related_columns) - {related_column})
                                    used_dimensions.append(related_column)
                                    # import pdb;pdb.set_trace()
                                    if df_filtered.shape[0] == 1:  # 当通过部分筛选条件可得到单个指标时，直接返回
                                        break

                            # 投诉流程 取 空值 作为默认值
                            if "投诉流程" in related_columns and df_filtered.shape[0] > 1 and \
                                    "投诉流程" not in used_dimensions:
                                df_filtered_tmp = df_filtered[df_filtered["投诉流程"].isin(["整体投诉"])].copy()
                                if df_filtered_tmp.shape[0] > 0:
                                    df_filtered = df_filtered_tmp.copy()
                            # 指标保单类型 取 空值 作为默认值
                            if "指标保单类型" in related_columns and df_filtered.shape[0] > 1 and \
                                    "指标保单类型" not in used_dimensions:
                                df_filtered_tmp = df_filtered[df_filtered["指标保单类型"].isna()].copy()
                                if df_filtered_tmp.shape[0] > 0:
                                    df_filtered = df_filtered_tmp.copy()
                            # 投保环节 取 承保 作为默认值
                            if "投保环节" in related_columns and df_filtered.shape[0] > 1 and \
                                    "投保环节" not in used_dimensions:
                                df_filtered_tmp = df_filtered[df_filtered["投保环节"] == "承保"].copy()
                                if df_filtered_tmp.shape[0] > 0:
                                    df_filtered = df_filtered_tmp.copy()

                            # +++++++++++++++++++++++++++++++++++++++++++++++
                            # 寻找统计口径
                            # 如果没有统计口径，通过语义找统计口径
                            ner_time_result = jio.ner.extract_time(user_question, time_base=time.time())
                            touch_time_result = re.findall(r'(?:日|号|天|月|季|年)', user_question)
                            ner_time_result = ner_time_result or touch_time_result
                            if "统计口径" in related_columns:
                                # 去重
                                candidates_original = list(set(df_filtered["统计口径"]))
                                # 删除空值
                                candidates_original = [value if value == value else ""
                                                       for value in candidates_original]
                                new_question = copy.deepcopy(user_question)

                                if ner_time_result:  #如果句子中有统计口径的词
                                    # 只保留当月作为默认值
                                    candidates = [value for value in candidates_original if value in ["当月"]]

                                    # 临时解决方案，需要训练嵌入模型
                                    for value in cls.day_values:
                                        user_question = user_question.replace(value, "")
                                    if "累计" in user_question or re.findall(r'(?:日|号|天)', user_question):
                                        added = False
                                        if "累计" in user_question:
                                            if "月" in user_question and "年" in user_question and "月累计" in candidates_original:
                                                candidates += ["月累计"]
                                                added = True
                                            elif "月" not in user_question and "年" in user_question and "年累计" in candidates_original:
                                                candidates += ["年累计"]
                                                added = True
                                            elif "月" in user_question and "月累计" in candidates_original:
                                                candidates += ["月累计"]
                                                added = True
                                            elif re.findall(r'(?:日|号|天)', user_question) and "月累计" in candidates_original:
                                                candidates += ["月累计"]
                                                added = True
                                            elif re.findall(r'(?:日|号|天)', user_question) and "年累计" in candidates_original:
                                                candidates += ["年累计"]
                                                added = True

                                        for stat in candidates_original:
                                            if "累计" in stat and stat in user_question:
                                                candidates += [stat]
                                                added = True

                                        if re.findall(r'(?:日|号|天)', user_question) and "当日" in candidates_original:
                                            candidates += ["当日"]
                                            added = True
                                            new_question = re.sub(r'([本月|当月|下月|下下月])', '', new_question)

                                        if re.findall(r'(?:宽一|应交月上月|下月累计|下月)', user_question) and \
                                                "下月" in candidates_original:
                                            candidates += ["下月"]
                                            added = True

                                        if re.findall(r'(?:应交月当月|下下月累计|下下月)', user_question) and \
                                                "下下月" in candidates_original:
                                            candidates += ["下下月"]
                                            added = True

                                        if not added:
                                            candidates = copy.deepcopy(candidates_original)
                                    else:
                                        if "统计口径" in new_index_vals:
                                            candidates += new_index_vals["统计口径"]

                                else:  # 句子中没有提到统计口径，默认返回当月？ #1017修改
                                    all_candidates = cls.index_split[cls.index_split["元子指标名称"].isin(df_filtered['元子指标名称'].tolist())]['统计口径'].tolist()
                                    all_candidates = list(set(all_candidates))
                                    if len(all_candidates) == 1:
                                        candidates = all_candidates
                                    elif '当月' in all_candidates:
                                        candidates = ['当月']
                                    else:
                                        candidates = candidates_original

                                candidates = [item for item in candidates if item in candidates_original]
                                if candidates:
                                    candidates = list(set(candidates))
                                    logger.info(f"[待匹配问题]{new_question}")
                                    logger.info(f"[指标还原候选列表]{candidates}")
                                    matched_result1 = await FuzzyMatch.get_embedding_cosine_sim(
                                        new_question, candidates
                                    )
                                    matched_result2 = await FuzzyMatch.get_embedding_cosine_sim(
                                        new_question.replace(question_name, ""), candidates
                                    )
                                    stat_caliber_embedding1 = matched_result1[0]['word']
                                    stat_caliber_embedding2 = matched_result2[0]['word']
                                    if stat_caliber_embedding1 == stat_caliber_embedding2:
                                        logger.info(f"[指标还原匹配结果]2次匹配结果相同")
                                        stat_caliber_embedding = stat_caliber_embedding1
                                    else:
                                        logger.info(f"[指标还原匹配结果]2次匹配结果不同：{stat_caliber_embedding1}，"
                                                  f"{stat_caliber_embedding2}")
                                        if "当日" in [stat_caliber_embedding1, stat_caliber_embedding2]:
                                            stat_caliber_embedding = "当日"
                                        else:
                                            stat_caliber_embedding = stat_caliber_embedding1
                                    logger.info(f"[指标还原匹配结果]{stat_caliber_embedding}")
                                    if stat_caliber_embedding:
                                        df_filtered = df_filtered[df_filtered["统计口径"] == stat_caliber_embedding].copy()
                                    else:
                                        df_filtered = df_filtered[df_filtered["统计口径"].isna()].copy()

                            # +++++++++++++++++++++++++++++++++++++++++++++++

                            # print("================debug===============")
                            # print(df_filtered.head())

                            df_filtered = df_filtered[["元子指标名称", "捷报名称", "中台中文名称"]].drop_duplicates()
                            basic_index_names = list(df_filtered["元子指标名称"])
                            news_index_names = list(df_filtered["捷报名称"])
                            standard_index_names = list(df_filtered["中台中文名称"])
                        else:
                            logger.info(f"根据筛选条件未找到中台中文名称！")
                            basic_index_names = []
                            news_index_names = []
                            standard_index_names = []
                    else:
                        # 客户域或有指标编码（经过主题指标填补）
                        time_dimension = cls._find_frequency_keyword(user_question)
                        if topic_filling:
                            # 有指标编码（经过主题指标填补）
                            df_filtered = cls.index_split[(cls.index_split["中台指标编码"].isin(index_code)) &
                                                          (cls.index_split["意图"] == table_name)].copy()
                        else:
                            # 客户域
                            df_filtered = cls.index_split[cls.index_split["中台中文名称"].isin(basic_index_name)].copy()

                        df_filtered = df_filtered[["元子指标名称", "捷报名称", "中台中文名称"]].drop_duplicates()
                        basic_index_names = list(df_filtered["元子指标名称"])
                        news_index_names = list(df_filtered["捷报名称"])
                        standard_index_names = list(df_filtered["中台中文名称"])

                    # 统计指标维度乘积
                    # {'服务渠道': ['临柜服务'], '客户类型': ['全量', '剔除老年人'], '投诉流程': ['监管全量']}
                    index_val_product = 1
                    for key, value in index_vals_product.items():
                        index_val_product *= len(value)

                    # print("new index names:", news_index_names)
                    # print("index val product:", index_val_product)
                    # print("standard index names:", standard_index_names)
                    # print("topic filling:", topic_filling)
                    

                    # 汇集该指标的所有可能性
                    # 如果指标列表长度==指标维度长度乘积，则作为多指标整体输出
                    # 否则作为多可能性输出
                    # TODO: 多可能性的判定标准需要修改
                    multiple_possibility = True
                    if (len(news_index_names) == index_val_product and index_val_product >= 2) or topic_filling:
                        multiple_possibility = False
                        index_entry_updated = copy.deepcopy(entry)

                        table_models = []
                        for news_index_name, standard_index_name in zip(news_index_names, standard_index_names):
                            # 匹配表模型
                            freq = cls._find_frequency_keyword(user_question, "月", standard_index_name)
                            if freq not in ["日", "月"]:
                                freq = "月"
                            if table_name in ["通用", "产品", "绩优", "核保", "业绩大宽表"]:
                                table_model_key = f"{table_name}意图-{freq}频"
                            else:
                                table_model_key = f"{table_name}意图"
                            table_models.append(cls.table_model_dict[table_model_key])
                        # 如果表模型不一致，则作为多可能性处理
                        if len(set(table_models)) >= 2:
                            multiple_possibility = True

                        # 构造候选条目
                        index_entry_updated.update({
                            'indexName': basic_index_names,
                            'indexNameNews': news_index_names,
                            'indexNameStandard': standard_index_names,
                            'indexCode': [cls.middle_index_name_code_dict[table_name + name]
                                            for name in standard_index_names],
                            'tableModel': table_models[0],
                            'timeDimension': time_dimension,
                        })
                        possibilities_index_entry.append(index_entry_updated)

                    if multiple_possibility:
                        for basic_index_name, news_index_name, standard_index_name \
                                in zip(basic_index_names, news_index_names, standard_index_names):
                            index_entry_updated = copy.deepcopy(index_entry)

                            # 匹配表模型
                            freq = cls._find_frequency_keyword(user_question, "月", standard_index_name)
                            if freq not in ["日", "月"]:
                                freq = "月"
                            if table_name in ["通用", "产品", "绩优", "核保", "业绩大宽表"]:
                                table_model_key = f"{table_name}意图-{freq}频"
                            else:
                                table_model_key = f"{table_name}意图"
                            table_model = cls.table_model_dict[table_model_key]

                            index_entry_updated.update(
                                {
                                    'indexName': [basic_index_name],
                                    'indexNameNews': [news_index_name],
                                    'indexNameStandard': [standard_index_name],
                                    'indexCode': [cls.middle_index_name_code_dict[table_name + standard_index_name]],
                                    'tableModel': table_model,
                                    'timeDimension': time_dimension,
                                }
                            )
                            possibilities_index_entry.append(index_entry_updated)

                if possibilities_index_entry:
                    possibilities_index_result[question_name] = possibilities_index_entry

            # 组合所有可能性（笛卡尔积）
            candidates_index_result = []
            try:
                lists = [v for _, v in possibilities_index_result.items()]
                for comb in itertools.product(*lists):
                    # 转换成每个 question_name -> [entry] 的格式
                    converted = dict(zip(list(possibilities_index_result.keys()), [[el] for el in comb]))
                    candidates_index_result.append(converted)

                for new_index_result in candidates_index_result:
                    tmp_attr_result = {
                        'attrVal': new_attr_vals,
                        'attrObj': new_attr_objs,
                        'indexResult': new_index_result,
                        'llmAnswer': Util.read_map_value(attr_result, 'llmAnswer', {})
                    }

                    tmp_table_name = list(new_index_result.values())[0][0].get('tableName')
                    valid_table_names.append(tmp_table_name)
                    news_indices = '，'.join(list(new_index_result.values())[0][0].get('indexNameNews', []))
                    std_indices = '，'.join(list(new_index_result.values())[0][0].get('indexNameStandard', []))

                    choice = f"{news_indices}"
                    hidden = f"{news_indices}|{std_indices}" if set(valid_table_scopes) != {'客户'} else f"{std_indices}"
                    hidden = StringUtil.get_cleaned_question(hidden) if hasattr(StringUtil, 'get_cleaned_question') else hidden

                    ret.append(tmp_attr_result)
                    question_info_list.append({'questions': choice, 'hiddenQuestions': hidden})
                    other_info_list.append({'choice': choice, 'hidden': hidden, 'attrResult': [tmp_attr_result], 'attrResult4Time': [attr_result]})
            except IndexError as e:
                logger.error(f'指标还原为空：{e}')
                logger.info(f'指标还原为空：{candidates_index_result}')

        # 过滤并去重按 valid_table_scopes
        valid_table_names1 = [n for n in valid_table_names if n not in ['产品', '绩优', '通用']]
        valid_table_names2 = [n for n in valid_table_names if n in ['产品', '绩优', '通用']]
        # 保持优先级
        valid_table_names2 = await cls._prior_elements(valid_table_names2) if hasattr(cls, '_prior_elements') else valid_table_names2
        valid_table_names_final = valid_table_names1 + valid_table_names2

        new_ret, new_question_info_list = [], []
        for tmp_ret, question_info, other_info in zip(ret, question_info_list, other_info_list):
            table_name = list(tmp_ret['indexResult'].values())[0][0].get('tableName')
            if table_name in valid_table_names_final:
                tmp_flag = False
                if question_info not in new_question_info_list:
                    tmp_flag = True
                if tmp_ret not in new_ret and tmp_flag:
                    # 去重 indexCode
                    new_index_result = tmp_ret['indexResult']
                    deduplicate_index_result = {}
                    all_index_code_list = []
                    for qn, idx_list in new_index_result.items():
                        current_index_dict = idx_list[0]
                        current_index_code = set(current_index_dict.get('indexCode', []))
                        if current_index_code not in all_index_code_list:
                            all_index_code_list.append(current_index_code)
                            deduplicate_index_result.update({qn: idx_list})
                    tmp_ret['indexResult'] = deduplicate_index_result
                    new_ret.append(tmp_ret)
                    new_question_info_list.append(question_info)
                    data_dict[other_info['hidden']] = {
                        'attrResult': other_info['attrResult'], 
                        'attrResult4Time': other_info['attrResult4Time']
                        }

        record = {'isCustomer': '非客户域' if set(valid_table_scopes) != {'客户'} else '客户域', 
                  'userQuestion': state.question, 
                  'data': data_dict}

        return new_ret, new_question_info_list, record
    
    @classmethod
    async def _get_topic_index(cls, current_topic, current_intention):
        """Return list of index codes for a topic (uses cls.df_topic if available)."""
        if hasattr(cls, 'df_topic') and isinstance(cls.df_topic, pd.DataFrame) and not cls.df_topic.empty:
            return list(set(cls.df_topic[cls.df_topic["topic"] == current_topic]["index_code"]))
        return []

    # HIGH PRIORITY 指标-4
    @classmethod
    async def topic_recognize(cls, state: PipelineState, attr_results: List[Dict]):
        """主题识别

        background:
            比如主题通过维度识别，未识别指标时，通过主题表带出主题的L0指标作为识别的指标。
            问题中有主题，但没指标的情况，进行填充

        solution:
            rewrite from topic_analysis.py: reformat_topic

        Args:
            state(PipelineState): 当前pipeline状态
            attr_results(List[Dict]): 维度识别结果

        Returns:
            List[Dict]: 新的维度识别结果
            Bool: 是否进行主题填充
        """
        process_id = state.request_id
        special_values = ['监管首次转办', '监管三次及以上转办', '监管二次转办']
        for items in attr_results:
            if len(items.get("indexResult", {})) == 0:
                special_words = [word_item['word'] for item in items.get('attrVal', []) for word_item in item.get('word_list', []) if
                                 word_item['word'] in special_values]
                if special_words:
                    items["indexResult"] = {'案件量': [{'indexName': ['案件量'], 'indexCode': ['案件量'], 'tableName': '客服', 'msg': '补充指标'}]}

        new_attribute_results = []
        topic_filling = False
        special_dim_dict = {"监管首次投诉":{'intent':'监管首次投诉','dims':["客户类型","投诉流转二层","投诉流转三层","投诉流转四层"]},
                            "监管直达转办":{'intent':'监管直达投诉','dims':["投诉流转二层"]},
                            "监管直达引导":{'intent':'监管直达投诉','dims':["投诉流转三层"]},
                            "业绩客户":{'intent':'NBEV拆解全景图-过程客户','dims':['客户分类','新老客标签']},
                            "业绩代理人":{'intent':'业绩大宽表','dims':['绩优类型整体','绩优类型大类','绩优类型一类']},
                            "业绩产品":{'intent':'业绩大宽表','dims':['产品推动险种组合大类描述','产品推动险种组合一级描述']}}

        for attr_result in attr_results:
            attr_values = Util.read_map_value(attr_result, "attrVal", [])
            attr_objs = Util.read_map_value(attr_result, "attrObj", [])
            index_result = Util.read_map_value(attr_result, "indexResult", {})

            new_attr_values = []
            topic_attr_values = []
            tmp_topics = []

            special_flag = False
            for attr_val in attr_values:
                columns_name_list = [word_entry.get("columns_name") for word_entry in attr_val.get("word_list", [])]
                if "主题" in columns_name_list:
                    other_word_list = [word_entry for word_entry in attr_val.get("word_list", []) 
                                       if word_entry.get("columns_name") != "主题"]
                    topic_word_list = [word_entry for word_entry in attr_val.get("word_list", []) 
                                       if word_entry.get("columns_name") == "主题"]
                    if not index_result:
                        final_word_list = topic_word_list
                    else:
                        final_word_list = other_word_list if other_word_list else attr_val.get("word_list", [])
                else:
                    final_word_list = attr_val.get("word_list", [])

                new_word_list = []
                for word_entry in final_word_list:
                    if (word_entry.get("columns_name") == "主题") and \
                        (word_entry.get("operator") == "in") and \
                            (word_entry.get("word") not in tmp_topics):
                        word_entry.update({"org_word": attr_val.get("org_word")})
                        topic_attr_values.append(word_entry)
                        tmp_topics.append(word_entry.get("word"))
                    else:
                        new_word_list.append(word_entry)

                    if not index_result:
                        for key, specials in special_dim_dict.items():
                            if word_entry.get("columns_name") in specials["dims"] and word_entry.get("table_name") == specials["intent"]:
                                if key not in tmp_topics:
                                    special_flag = True
                                    cur_entry = {'org_word':key,'word':key, 'columns': 'model_topic', 'columns_name': '主题', 'table_name': word_entry.get("table_name"), 'operator': 'in'}
                                    topic_attr_values.append(cur_entry)
                                    tmp_topics.append(key)

                if new_word_list:
                    attr_val["word_list"] = new_word_list
                    new_attr_values.append(attr_val)
            attr_result.update({"attrVal": new_attr_values})

            for atr_obj in attr_objs:
                for key, specials in special_dim_dict.items():
                    if key in ["业绩产品"]:
                        for word_entry in atr_obj.get("word_list", []):
                            if word_entry.get("word") in specials["dims"] and word_entry.get("table_name") == specials["intent"]:
                                if len(tmp_topics) > 1 and (key in tmp_topics):
                                    tmp_topics = [key]
                                    topic_attr_values = [item for item in topic_attr_values if item.get('word') == key]
                                    Util.info(f"通过特定维度选择主题{key}")
                                    break

            new_attr_objs = []
            for attr_obj in attr_objs:
                new_word_list = []
                for word_entry in attr_obj.get("word_list", []):
                    if word_entry.get('word') != "主题":
                        new_word_list.append(word_entry)

                if new_word_list:
                    attr_obj["word_list"] = new_word_list
                    new_attr_objs.append(attr_obj)
            attr_result.update({"attrObj": new_attr_objs})

            if {"监管直达转办","监管首次投诉"}.issubset(set(tmp_topics)):
                topic_attr_values = [item for item in topic_attr_values if item.get('word') != '监管首次投诉']

            check_list = [['业绩代理人','业绩客户'],['业绩代理人','业绩产品']]
            for cur_list in check_list:
                if set(cur_list).issubset(set(tmp_topics)):
                    if special_flag:
                        topic_attr_values = [item for item in topic_attr_values if item.get('word') != cur_list[0]]
                    else:
                        topic_attr_values = [item for item in topic_attr_values if item.get('word') == cur_list[0]]

            if not index_result:
                topic_filling = True
                for topic_attr_value in topic_attr_values:
                    current_org_word = topic_attr_value.get("org_word")
                    current_topic = topic_attr_value.get("word")
                    current_intention = topic_attr_value.get("table_name")

                    index_code_list = await cls._get_topic_index(current_topic, current_intention)
                    if index_code_list and process_id in [4,5,6]:
                        index_code_list = [index_code_list[0]]
                    index_tmp = cls.index_split[(cls.index_split["中台指标编码"].isin(index_code_list)) & (cls.index_split["意图"] == current_intention)].copy()

                    if index_code_list and index_tmp.shape[0] >= 1:
                        index_result = {
                            current_org_word: [
                                {
                                    'indexName': list(index_tmp["元子指标名称"]),
                                    'indexCode': list(index_tmp["中台指标编码"]),
                                    'indexNameNews': list(index_tmp["捷报名称"]),
                                    'indexNameStandard': list(index_tmp["中台中文名称"]),
                                    'tableName': current_intention,
                                    'msg': f'识别到主题分析意图，“{current_org_word}”匹配到主题“{current_topic}”',
                                    'topic': [current_topic],
                                    'ifTopic': 1 if special_flag else 2
                                }
                            ]
                        }

                        attr_result.update({"indexResult": index_result})
                        new_attribute_results.append(copy.deepcopy(attr_result))

            else:
                if topic_attr_values:
                    all_topic_index = []
                    for topic_attr_value in topic_attr_values:
                        current_org_word = topic_attr_value.get("org_word")
                        current_topic = topic_attr_value.get("word")
                        current_intention = topic_attr_value.get("table_name")

                        topic_index_code_list = []
                        if hasattr(cls, 'df_topic') and not cls.df_topic.empty:
                            topic_index_code_list = list(set(cls.df_topic[cls.df_topic["topic"] == current_topic]["index_code"]))

                        index_name_list = []

                        if (current_intention, topic_index_code_list) not in all_topic_index:
                            all_topic_index.append((current_intention, topic_index_code_list))

                            for key, values in index_result.items():
                                for value in values:
                                    index_name_list += value.get('indexName', [])
                            if current_intention == "客户":
                                topic_filling = True
                                df_tmp = cls.index_split[(cls.index_split["中台指标编码"].isin(topic_index_code_list)) & (cls.index_split["意图"] == current_intention) & (cls.index_split["中台中文名称"].isin(index_name_list))].copy()
                            else:
                                df_tmp = cls.index_split[(cls.index_split["中台指标编码"].isin(topic_index_code_list)) & (cls.index_split["意图"] == current_intention) & (cls.index_split["元子指标名称"].isin(index_name_list))].copy()

                            if df_tmp.shape[0] >= 1:
                                index_result = {
                                    current_org_word: [
                                        {
                                            'indexName': list(df_tmp["元子指标名称"]),
                                            'indexCode': list(df_tmp["中台指标编码"]),
                                            'indexNameNews': list(df_tmp["捷报名称"]),
                                            'indexNameStandard': list(df_tmp["中台中文名称"]),
                                            'tableName': current_intention,
                                            'msg': f'识别到主题分析意图，“{current_org_word}”匹配到主题“{current_topic}”',
                                            'topic': [current_topic],
                                            'ifTopic': 1
                                        }
                                    ]
                                }
                                attr_result.update({"indexResult": index_result})
                                new_attribute_results.append(copy.deepcopy(attr_result))
                            else:
                                new_attribute_results.append(attr_result)
                else:
                    new_attribute_results.append(copy.deepcopy(attr_result))

        return new_attribute_results, topic_filling
    
    async def index_predict(self, attrs: List) -> List:
        """预测指标
        
        background:
            比如当涉及预测时，通过改写指标实现识别。目前直接硬改写（只支持 NBEV 预测）
        
        solution:
            1. rewrite from index_extraction.py: predict_index
            2. 新框架已支持；测试确认
        
        Args:
            attrs(List): 指标列表

        Returns:
            List: 改写后的指标列表
        """
        return []
    
    # HIGH PRIORITY 指标-6
    @classmethod
    async def dim_support_check(cls, attr) -> Dict:
        """维度不支持判断

        background:
            比如某些追表不支持查某些维度，需要提前拦截
        
        solution:
            1. learn from index_extraction.py: index_support_dim_check
            2. 新框架已支持，具体配置细节@颍楠
            3. 与 投诉相关场景 —— 维值之间有组合限定 一起做
        
        Args:
            attrs(List): 指标列表

        Returns:
            Dict: 改写后的指标
        """
        if not attr:
            return {}

        current_attr = attr[0]
        all_indexes = []

        for index_org, index_list in current_attr.get('indexResult', {}).items():
            index_name_news = [cur_item for item in index_list for cur_item in item.get('indexNameNews', [])]
            all_indexes.extend(index_name_news)

        all_columns = []

        for item in current_attr.get('attrVal', []):
            for word_item in item.get('word_list', []):
                columns = word_item.get('columns', "")
                columns_name = word_item.get('columns_name', "")
                if columns.startswith("user_"):
                    all_columns.append(columns_name)

        for item in current_attr.get('attrObj', []):
            for word_item in item.get('word_list', []):
                columns = word_item.get('columns', "")
                word = word_item.get('word')
                if columns.startswith("user_"):
                    all_columns.append(word)

        df_filter = cls.index_split[cls.index_split["捷报名称"].isin(all_indexes)][["捷报名称", "指标不支持的维度"]].copy()
        df_filter = df_filter.dropna()
        df_filter = df_filter.drop_duplicates()

        if df_filter.empty:
            return {}

        ref_dict = dict(zip(df_filter["捷报名称"], df_filter["指标不支持的维度"]))

        ret_dict = {}
        all_unsupport = []
        for index_name in all_indexes:
            if index_name in ref_dict:
                uspts = ref_dict[index_name]
                uspts = uspts.split(";")
                uspt = set(uspts).intersection(set(all_columns))
                uspt = list(uspt)
                ref_dict[index_name] = uspt
                all_unsupport.extend(uspt)
            else:
                ret_dict[index_name] = []

        if all_unsupport:
            return ref_dict

        return {}
    
    async def dim_priority_check(self, question: str) -> str:
        """维度重合时的表优先级

        solution:
            1.后置模块
            2.rewrite from affix_search.py: search_words

        """
        return question
    
    async def format_convert(self, question: str) -> str:
        """结果类型
        
        background:
            返回的结果有：未识别指标、维度不支持、二次确认等

        solution:
            1. learn from restful.py: direct_chat
        """
        return question
    
    async def time_info_recognize(self, question: str) -> str:
        """时间类型等识别
        
        background:
            获取时间类型识别，时间频率

        solution:
            1. learn from time_extraction_dev.py
            2. 跟@德劲确认
        """
        return question
    
    # MARK: 捷报
    @classmethod
    def _jb_data_input_type_convert_recognize(cls, question: str) -> str:
        dummy_filed = ['query_number', 'analysis']
        return "other"

    @classmethod
    async def jb_data_input_convert(cls, question: str) -> str:
        dummy_question_type = cls._jb_data_input_type_convert_recognize(question)
        return question


if __name__ == "__main__":
    def _build_debug_attr_results():
        table_name = "个险_投诉日报_按受理时间"
        index_name = "监管投诉引导率"

        return [
            {
                "attrVal": [
                    {
                        "org_word": "其他",
                        "word_list": [
                            {
                                "word": "其他",
                                "columns": "user_accept_method_desc",
                                "columns_name": "受理方式描述(投诉方式)",
                                "table_name": table_name,
                                "operator": "in",
                            },
                            {
                                "word": "其他",
                                "columns": "user_business_src_desc",
                                "columns_name": "业务来源描述",
                                "table_name": table_name,
                                "operator": "in",
                            },
                        ],
                    }
                ],
                "attrObj": [],
                "indexResult": {
                    index_name: [
                        {
                            "indexName": [index_name],
                            "indexCode": [index_name],
                            "tableName": table_name,
                            "msg": f"指标“{index_name}”定义清晰，匹配到“{index_name}”",
                        }
                    ]
                },
                "llmAnswer": {"问题类型": "寿险", "句子拆解": []},
            },
            {
                "attrVal": [
                    {
                        "org_word": "其他",
                        "word_list": [
                            {
                                "word": "其他",
                                "columns": "user_accept_source_desc",
                                "columns_name": "受理来源描述",
                                "table_name": table_name,
                                "operator": "in",
                            }
                        ],
                    }
                ],
                "attrObj": [],
                "indexResult": {
                    index_name: [
                        {
                            "indexName": [index_name],
                            "indexCode": [index_name],
                            "tableName": table_name,
                            "msg": f"指标“{index_name}”定义清晰，匹配到“{index_name}”",
                        }
                    ]
                },
                "llmAnswer": {"问题类型": "寿险", "句子拆解": []},
            },
        ]

    def _build_debug_state():
        return PipelineState(
            request_id="debug-request",
            question="其他长期寿险-非获客类的监管投诉引导率贡献率。",
            database_id="debug-db",
        )

    async def _main():
        PreConverter.init()
        state = _build_debug_state()
        attribute_results = _build_debug_attr_results()

        print("[before convert]", state.question)
        state = await PreConverter.execute(state)
        print("[after convert]", state.question)

        # index_convert was already invoked inside execute; no need to call it again here.

        class DataLoader:
            table_names = ['捷报_寿险整体', '捷报_个险', '捷报_银保', '捷报_网格', '捷报_新渠道', '客户']

        print("[sample attr_results count]", len(attribute_results))
        print("[sample attr_results[0].indexResult keys]", list(attribute_results[0]["indexResult"].keys()))

        ret = await PreConverter.index_resolve(
            state=state, attr_results = attribute_results,
            valid_table_scopes=list(
                set(DataLoader.table_names) - 
                set(['捷报_寿险整体', '捷报_个险', '捷报_银保', '捷报_网格', '捷报_新渠道']) - {"客户"}
                )
            )
        print("[index_resolve outputs]", ret)

    asyncio.run(_main())