# Copyright 2026 The LLaMA Factory Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writing dimensions for the atom-diversity pilot.

Each generated sentence samples one value per enabled dimension. Values only steer
wording; they must never change the cell's intent or arguments.
"""

SCENES = (
    "展会展台接待",
    "商场导览",
    "酒店大堂",
    "银行营业厅",
    "学校科技课",
    "家里客厅",
    "公司前台",
    "博物馆讲解",
    "发布会舞台",
    "医院导诊台",
    "餐厅门口迎宾",
    "实验室调试",
)

PERSONAS = (
    "好奇的小学生",
    "带孩子的家长",
    "不太懂科技的老人",
    "赶时间的上班族",
    "外地来的游客",
    "展台工作人员",
    "调试机器人的工程师",
    "活动主持人",
    "爱开玩笑的年轻人",
    "礼貌的商务客人",
    "第一次见机器人、有点紧张的人",
    "店里的服务员",
)

STYLES = (
    "直接命令",
    "礼貌请求（用“请”“麻烦你”，不用问句）",
    "随口口语",
    "带理由或场景说明",
    "商量、建议的语气（如“要不……吧”“咱……吧”）",
    "省略主语的短句",
    "先铺垫一句再提要求",
)

# Asking for information: questions are the natural form, unlike actions.
QUERY_STYLES = (
    "直接问",
    "请它帮忙查（如“帮我查下”“麻烦看看”）",
    "用“能不能、可以、你知道……吗”的礼貌问法",
    "随口口语",
    "带理由或场景说明",
    "省略主语的短句",
    "先铺垫一句再问",
)

# Label-preserving noise only: corrections or extra requests would change the label.
NOISES = (
    "带语气词和停顿（如嗯、那个、呃）",
    "有一两个重复的词",
    "语音转写的同音错别字，只能出现在不影响意思的字上",
    "一口气说完，没有标点",
)
NOISE_RATE = 0.4

# Action phrasing per cell. Every phrase must still pass the wording rules in
# planning_atoms.derive_human_action (forward needs 前进/往前/向前/朝前, turns need 转/旋, ...).
_FWD = ("往前走", "往前迈", "往前挪", "向前移动", "朝前走", "前进", "往前来", "向前跨")
_BACK = ("往后退", "后退", "倒退", "往后挪", "退后", "向后移动", "朝后走", "往后撤")


def _side(side: str) -> tuple[str, ...]:
    return (f"向{side}平移", f"往{side}边挪", f"往{side}横着走", f"向{side}侧移", f"往{side}靠", f"{side}移",
            f"往{side}边让一让")


def _turn(side: str) -> tuple[str, ...]:
    return (f"向{side}转", f"{side}转", f"往{side}转一下", f"转向{side}边", f"朝{side}边转过去", f"身体往{side}转")


def _turn_deg(side: str) -> tuple[str, ...]:
    return (f"向{side}转", f"{side}转", f"往{side}边转", f"朝{side}转过去", f"身体往{side}转", f"往{side}旋转")


def _turn_bit(side: str) -> tuple[str, ...]:
    return (f"稍微往{side}转一点", f"向{side}转一点", f"稍微{side}转", f"往{side}转一些", f"轻轻往{side}转",
            f"往{side}微微转一点")


KERNELS = {
    "sit_down": ("坐下", "坐一会儿", "坐着", "坐下来歇歇", "找个地方坐", "坐好"),
    "lie_down": ("躺下", "躺一会儿", "平躺", "躺平", "躺着休息", "躺倒"),
    "stand_up_casual": ("站起来", "起身", "起立", "站好", "站起身来", "立正"),
    "fwd_plain": _FWD, "fwd_steps": _FWD, "fwd_little": _FWD,
    "back_plain": _BACK, "back_steps": _BACK, "back_little": _BACK,
    "left_plain": _side("左"), "left_steps": _side("左"),
    "right_plain": _side("右"), "right_steps": _side("右"),
    "turn_left": _turn("左"), "turn_right": _turn("右"),
    "turn_left_deg": _turn_deg("左"), "turn_right_deg": _turn_deg("右"),
    "turn_left_bit": _turn_bit("左"), "turn_right_bit": _turn_bit("右"),
    "turn_around": ("转身", "向后转", "转过身去", "掉头转过去", "转过去背对着我", "往后转"),
    "spin": ("转一圈", "原地转个圈", "转个圈圈", "旋转一圈", "自己转一圈"),
    "spin_right": ("向右转一圈", "往右转个圈", "朝右边旋转一圈", "右转一整圈"),
}

# Label convention (user decision 2026-10-06): "转一下" is a plain 90° turn;
# only 一点/稍微/一些/微微 mean a small 15° turn.
HINT_OVERRIDES = {
    # The original hint's example "别坐着了" makes the rule validator see both 坐 and 站.
    "stand_up_casual": "侧重口语或带场景理由的说法；句子里不要出现“坐”“躺”字",
    "turn_left": "只说转向，不说角度；可以说“转一下”，不出现“一点、稍微、一些、一圈、转身、后面”",
    "turn_right": "只说转向，不说角度；可以说“转一下”，不出现“一点、稍微、一些、一圈、转身、后面”",
    "turn_left_bit": "用“一点、稍微、一些、微微”这类程度词，不说角度，不用“一下”表示程度",
    "turn_right_bit": "用“一点、稍微、一些、微微”这类程度词，不说角度，不用“一下”表示程度",
    # User decision 2026-10-06: a bare farewell to the robot plans a wave.
    "wave_scene": "对它道别，或请它跟人打招呼；约一半只道别、不提挥手（如“拜拜”“好的，回头见”），"
                  "另一半明说挥手；不要只说“你好”",
}

DIMENSIONS = ("scene", "persona", "style", "noise", "kernel")
