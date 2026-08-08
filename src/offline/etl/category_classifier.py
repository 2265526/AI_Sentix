"""
分级类目体系与自动分类

三级结构：大类（13 个）→ 中类 → 小类。入库自动分类：
  1. 有中文 breadcrumb（如 ["服装鞋包","女装","衬衫"]）：按路径段匹配类目树；
  2. 无 breadcrumb 或路径非中文（默认场景）：按商品名中文关键词分类；
  3. 都未命中：归入「其他/其他」。
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 一、三级类目种子树（大类 → 中类 → 小类）
CATEGORY_TREE: Dict[str, Dict[str, List[str]]] = {
    "服装鞋包": {
        "女装": ["衬衫", "T恤", "连衣裙", "牛仔裤", "裤装", "外套", "内衣家居服", "睡衣"],
        "男装": ["衬衫", "T恤", "牛仔裤", "裤装", "外套", "卫衣", "内衣袜"],
        "鞋靴": ["运动鞋", "凉鞋拖鞋", "高跟鞋", "靴子"],
        "箱包": ["双肩包", "单肩包", "钱包卡包", "旅行箱"],
        "配饰": ["帽子", "眼镜", "围巾手套", "腰带", "首饰", "手表", "头饰"],
    },
    "美妆个护": {
        "彩妆": ["口红", "眼影", "粉底", "眉笔", "美甲"],
        "护肤": ["面膜", "精华", "防晒", "洁面", "爽肤水", "面霜"],
        "香水": ["女士香水", "男士香水"],
        "洗护发": ["洗发水", "护发素", "发膜"],
        "身体护理": ["沐浴露", "身体乳", "除臭剂"],
        "美容工具": ["吹风机", "卷发棒", "化妆刷"],
        "男士护理": ["剃须刀", "须后水"],
    },
    "手机数码": {
        "手机": ["智能手机", "老年机"],
        "手机配件": ["手机壳", "充电器", "数据线", "电池", "屏幕保护膜"],
        "音频": ["耳机", "音箱", "麦克风", "功放"],
        "电脑": ["笔记本电脑", "显示器", "键盘", "鼠标", "显卡", "网络设备"],
        "相机": ["相机", "镜头", "无人机", "摄像头"],
        "智能穿戴": ["智能手表", "手环"],
    },
    "家电": {
        "大家电": ["冰箱", "洗衣机", "电视", "空调"],
        "厨房电器": ["电水壶", "榨汁机", "咖啡机", "净水器"],
        "生活电器": ["吸尘器", "风扇", "加湿器", "电吹风"],
    },
    "家居生活": {
        "家具": ["桌子", "椅子", "收纳架", "书架"],
        "家纺床品": ["床单", "被套", "枕头", "被子", "毛毯"],
        "厨房用品": ["锅具", "餐具", "刀具", "水杯", "保鲜盒", "围裙"],
        "灯具": ["台灯", "吊灯", "装饰灯"],
        "装饰": ["花瓶", "相框", "镜子", "窗帘"],
        "清洁用品": ["清洁剂", "拖把", "垃圾袋"],
        "园艺": ["花盆", "种子", "园艺工具"],
        "卫浴": ["毛巾", "浴室用品"],
        "派对用品": ["气球", "贺卡"],
    },
    "母婴玩具": {
        "婴童服饰": ["婴儿服", "儿童外套", "儿童睡衣"],
        "尿裤喂养": ["奶粉", "奶瓶", "纸尿裤", "婴儿食品"],
        "婴童洗护": ["婴儿沐浴", "婴儿护肤"],
        "玩具": ["娃娃", "积木", "毛绒玩具", "益智玩具", "模型玩具"],
        "童车座椅": ["婴儿车", "儿童座椅", "背带"],
        "孕妇用品": ["孕妇装", "孕妇营养"],
    },
    "运动户外": {
        "运动服饰": ["运动T恤", "运动裤", "骑行服", "运动外套"],
        "健身器材": ["哑铃", "瑜伽垫", "健身器械", "阻力带"],
        "户外装备": ["帐篷", "登山包", "雨衣", "遮阳伞"],
        "骑行": ["自行车", "骑行头盔", "骑行配件"],
        "球类运动": ["足球", "羽毛球", "篮球", "台球"],
    },
    "健康保健": {
        "个人护理": ["按摩仪", "电动牙刷", "牙膏", "口腔护理"],
        "营养保健": ["蛋白粉", "维生素", "膳食补充"],
        "医疗保健": ["体温计", "血压计", "急救用品"],
    },
    "食品饮料": {
        "零食": ["薯片", "坚果", "饼干", "糖果"],
        "饮料": ["矿泉水", "果汁", "茶", "碳酸饮料"],
        "乳品蛋": ["牛奶", "酸奶", "奶粉"],
        "酒类": ["啤酒", "葡萄酒", "烈酒"],
        "调味粮油": ["调味品", "酱料", "食用油"],
    },
    "宠物用品": {
        "宠物食品": ["猫粮", "狗粮", "宠物零食"],
        "猫狗用品": ["宠物玩具", "宠物窝", "宠物牵引"],
        "水族": ["鱼缸", "水族用品"],
        "宠物服饰": ["宠物衣服", "宠物配饰"],
    },
    "文具图书": {
        "书写工具": ["钢笔", "马克笔", "中性笔"],
        "本册纸品": ["笔记本", "便签", "纸张"],
        "办公用品": ["文件夹", "订书机", "胶带"],
        "图书": ["图书", "杂志"],
    },
    "汽车摩托配件": {
        "汽车配件": ["汽车零件", "车灯", "轮胎"],
        "汽车养护": ["车蜡", "洗车用品", "润滑油"],
        "摩托车": ["摩托车头盔", "摩托车配件", "摩托车保养"],
    },
    "爱好收藏": {
        "游戏": ["游戏机", "游戏卡", "游戏周边"],
        "手办收藏": ["手办", "模型", "收藏卡"],
        "乐器": ["吉他", "乐器配件"],
        "DIY手工": ["手工材料", "贴纸"],
    },
    "其他": {"其他": ["其他"]},
}

# 其他兜底
FALLBACK = ("其他", "其他", "其他/其他")


# 三、关键词 → (大类, 小类) 分类器（在线 CSV 无 breadcrumb 时使用）
KEYWORD_MAP: List[Tuple[List[str], str, str]] = [
    # 服装鞋包
    (["连衣裙", "裙子", "长裙", "短裙", "半身裙"], "服装鞋包", "女装"),
    (["衬衫", "衬衣", "blouse", "camisa"], "服装鞋包", "女装"),
    (["T恤", "t-shirt", "tee", "tank", "背心"], "服装鞋包", "男装"),
    (["牛仔裤", "jeans", "牛仔"], "服装鞋包", "男装"),
    (["裤子", "裤", "pants", "pantalon", "短裤", "shorts", "cargo"], "服装鞋包", "男装"),
    (["卫衣", "hoodie", "连帽"], "服装鞋包", "男装"),
    (["外套", "夹克", "jacket", "chamarra", "大衣", "风衣"], "服装鞋包", "男装"),
    (["睡衣", "pijama", "家居服", "浴袍"], "服装鞋包", "女装"),
    (["内衣", "内裤", "bra", "underwear", "袜子", "socks", "袜子"], "服装鞋包", "男装"),
    (["骑行服", "运动服", "运动装", "jersey"], "服装鞋包", "男装"),
    (["鞋", "zapatos", "sneaker", "sandal", "运动鞋", "凉鞋", "高跟鞋", "靴子", "拖鞋"], "服装鞋包", "鞋靴"),
    (["包", "背包", "mochila", "backpack", "双肩包", "单肩包", "钱包", "手袋", "挎包"], "服装鞋包", "箱包"),
    (["帽子", "sombrero", "cap", "hat", "鸭舌帽"], "服装鞋包", "配饰"),
    (["眼镜", "lentes", "墨镜", "gafas", "太阳镜"], "服装鞋包", "配饰"),
    (["围巾", "围脖", "手套", "guantes", "腰带", "cinturón"], "服装鞋包", "配饰"),
    (["手表", "reloj", "watch", "手链", "项链", "耳环", "戒指", "首饰"], "服装鞋包", "配饰"),
    # 美妆个护
    (["口红", "唇膏", "唇釉", "labial", "lipstick", "眼影", "sombra", "粉底", "base", "眉笔", "美甲"], "美妆个护", "彩妆"),
    (["面膜", "mascarilla", "mask", "精华", "serum", "防晒", "sunscreen", "洁面", "爽肤水", "toner", "面霜", "cream"], "美妆个护", "护肤"),
    (["香水", "perfume", "parfum", "eau de"], "美妆个护", "香水"),
    (["洗发", "shampoo", "shampo", "护发素", "conditioner", "发膜", "吹风机", "卷发棒", "hair dryer"], "美妆个护", "美容工具"),
    (["沐浴", "shower", "body wash", "沐浴露", "肥皂", "soap", "身体乳", "lotion", "除臭", "deodoran"], "美妆个护", "身体护理"),
    (["剃须", "shaving", "须后"], "美妆个护", "男士护理"),
    (["化妆刷", "brocha", "brush", "美妆蛋"], "美妆个护", "美容工具"),
    # 手机数码
    (["手机壳", "手机套", "case", "carcasa", "屏幕膜", "钢化膜"], "手机数码", "手机配件"),
    (["充电器", "charger", "数据线", "cable", "电池", "batería", "移动电源", "power bank"], "手机数码", "手机配件"),
    (["手机", "smartphone", "celular", "phone", "gadget", "walkie"], "手机数码", "手机"),
    (["耳机", "earphone", "headphone", "auricular", "耳塞", "airpods"], "手机数码", "音频"),
    (["音箱", "bocina", "speaker", "音响", "喇叭", "功放", "loa", "麦克风", "micrófono", "mic"], "手机数码", "音频"),
    (["笔记本", "laptop", "电脑", "computador", "monitor", "显示器", "键盘", "teclado", "keyboard", "鼠标", "mouse", "显卡", "网卡", "router"], "手机数码", "电脑"),
    (["相机", "cámara", "camera", "镜头", "lente", "镜头", "无人机", "drone", "摄像头", "监控", "CCTV"], "手机数码", "相机"),
    (["智能手表", "smartwatch", "手环", "fitness tracker", "穿戴"], "手机数码", "智能穿戴"),
    # 家电
    (["冰箱", "refrigerador", "洗衣机", "washing machine", "电视", "televisor", "tivi", "空调", "aire acondicionado"], "家电", "大家电"),
    (["电水壶", "kettle", "榨汁机", "咖啡机", "净水器", "purifier", "水壶"], "家电", "厨房电器"),
    (["吸尘器", "aspiradora", "风扇", "ventilador", "加湿器", "humidifier", "空气净化"], "家电", "生活电器"),
    # 家居生活
    (["床单", "sábana", "sheets", "被套", "枕套", "枕头", "almohada", "pillow", "被子", "edredón", "毛毯", "manta"], "家居生活", "家纺床品"),
    (["家具", "mueble", "furniture", "桌子", "mesa", "椅子", "silla", "收纳", "置物架", "rak", "书架"], "家居生活", "家具"),
    (["锅", "olla", "pot", "餐具", "碗", "盘子", "plato", "杯", "taza", "mug", "刀具", "cuchillo", "刀", "砧板", "保鲜盒"], "家居生活", "厨房用品"),
    (["灯", "lámpara", "lamp", "照明", "lighting"], "家居生活", "灯具"),
    (["花瓶", "florero", "相框", "marco", "photo frame", "镜子", "espejo", "mirror", "装饰", "decor", "窗帘", "cortina"], "家居生活", "装饰"),
    (["清洁剂", "清洁液", "limpiador", "拖把", "trapeador", "垃圾袋", "basura", "海绵", "esponja", "洗洁精", "detergen"], "家居生活", "清洁用品"),
    (["花盆", "maceta", "园艺", "jardín", "植物", "plant", "种子"], "家居生活", "园艺"),
    (["毛巾", "toalla", "towel", "浴巾", "浴帘"], "家居生活", "卫浴"),
    (["气球", "globo", "balón de fiesta", "派对", "fiesta", "贺卡", "tarjeta"], "家居生活", "派对用品"),
    # 母婴玩具
    (["玩具", "juguete", "toy", "积木", "bloque", "娃娃", "muñeca", "doll", "毛绒", "peluche", "拼图", "模型"], "母婴玩具", "玩具"),
    (["奶粉", "milk formula", "sữa bột", "奶瓶", "biberón", "纸尿裤", "pañal", "尿不湿", "婴儿食品"], "母婴玩具", "尿裤喂养"),
    (["婴儿", "bebé", "baby", "宝宝", "儿童", "niño", "kids", "童装", "童鞋"], "母婴玩具", "婴童服饰"),
    (["婴儿车", "cochecito", "stroller", "儿童座椅", "背带", "婴儿床"], "母婴玩具", "童车座椅"),
    (["孕妇", "maternidad", "maternity", "孕期"], "母婴玩具", "孕妇用品"),
    # 汽摩
    (["摩托", "moto", "motorcycle", "头盔", "casco", "车", "coche", "car", "汽车", "auto", "车灯", "轮胎", "llanta", "tire", "车蜡", "cera", "润滑油", "aceite motor", "刹车", "freno"], "汽车摩托配件", "摩托车"),
    # 运动户外
    (["健身", "fitness", "gym", "哑铃", "yoga", "瑜伽", "运动器械", "跑步机", "健身车"], "运动户外", "健身器材"),
    (["帐篷", "tienda de campaña", "tent", "露营", "camping", "登山", "徒步", "背包客", "雨衣", "imperm", "伞", "sombrilla", "umbrella"], "运动户外", "户外装备"),
    (["自行车", "bicicleta", "bike", "骑行", "ciclismo"], "运动户外", "骑行"),
    (["足球", "fútbol", "soccer", "羽毛球", "bádminton", "篮球", "baloncesto", "台球", "billar", "高尔夫", "golf", "网球", "乒乓球"], "运动户外", "球类运动"),
    (["运动", "deporte", "sport", "户外", "aire libre", "outdoor"], "运动户外", "运动服饰"),
    # 健康保健
    (["按摩", "masaje", "massage", "理疗", "电动牙刷", "cepill", "牙膏", "pasta", "toothpaste", "口腔", "牙刷"], "健康保健", "个人护理"),
    (["蛋白粉", "proteína", "维生素", "vitamin", "补充", "supplement", "营养"], "健康保健", "营养保健"),
    (["体温计", "termómetro", "血压计", "急救", "primeros auxilios", "first aid", "医疗", "médico"], "健康保健", "医疗保健"),
    # 食品饮料
    (["薯片", "薯条", "chips", "零食", "snack", "坚果", "nuez", "饼干", "galleta", "糖果", "dulce", "巧克力", "chocolate", "薯条"], "食品饮料", "零食"),
    (["水", "agua", "矿泉水", "饮料", "bebida", "果汁", "jugo", "jus", "茶", "碳酸", "soda", "汽水"], "食品饮料", "饮料"),
    (["牛奶", "leche", "milk", "酸奶", "yogur", "乳", "奶粉"], "食品饮料", "乳品蛋"),
    (["啤酒", "cerveza", "beer", "cider", "酒", "vino", "wine", "烈酒", "whisky"], "食品饮料", "酒类"),
    (["调味", "salsa", "sauce", "酱", "香料", "especia", "spice", "油", "aceite", "oil", "咖啡", "café", "coffee"], "食品饮料", "调味粮油"),
    # 宠物
    (["宠物", "mascota", "pet", "猫", "gato", "cat", "狗", "perro", "dog", "猫粮", "狗粮", "鱼缸", "acuario", "aquarium", "水族"], "宠物用品", "猫狗用品"),
    # 文具图书
    (["笔", "bolígrafo", "pen", "马克笔", "marker", "钢笔", "荧光笔", "笔记本", "cuaderno", "notebook", "便签", "胶带", "cinta", "tape", "文件夹", "carpeta", "订书机", "剪刀"], "文具图书", "书写工具"),
    (["图书", "libro", "book", "杂志", "revista", "magazine"], "文具图书", "图书"),
    # 爱好收藏
    (["游戏", "juego", "game", "gaming", "游戏机", "consola", "switch", "playstation", "xbox", "手办", "figura", "figurilla", "figure", "收藏", "coleccion", "collection", "模型", "modelo", "model", "乐器", "guitarra", "guitar", "吉他", "diy", "手工", "贴纸", "sticker", "pegatina", "徽章", "emblem"], "爱好收藏", "手办收藏"),
    # 其他兜底
    (["其他", "otros", "other", "lại", "khác"], "其他", "其他"),
]

# 小类关键词 → 大类内具体小类
SMALL_KEYWORD_MAP: Dict[str, Tuple[str, str]] = {    # 服装鞋包
    "衬衫": ("服装鞋包", "女装"), "衬衣": ("服装鞋包", "女装"), "连衣裙": ("服装鞋包", "女装"),
    "裙子": ("服装鞋包", "女装"), "T恤": ("服装鞋包", "男装"), "牛仔裤": ("服装鞋包", "男装"),
    "卫衣": ("服装鞋包", "男装"), "外套": ("服装鞋包", "男装"), "夹克": ("服装鞋包", "男装"),
    "运动鞋": ("服装鞋包", "鞋靴"), "凉鞋": ("服装鞋包", "鞋靴"), "拖鞋": ("服装鞋包", "鞋靴"),
    "高跟鞋": ("服装鞋包", "鞋靴"), "靴子": ("服装鞋包", "鞋靴"), "背包": ("服装鞋包", "箱包"),
    "双肩包": ("服装鞋包", "箱包"), "单肩包": ("服装鞋包", "箱包"), "钱包": ("服装鞋包", "箱包"),
    "帽子": ("服装鞋包", "配饰"), "眼镜": ("服装鞋包", "配饰"), "手表": ("服装鞋包", "配饰"),
    "项链": ("服装鞋包", "配饰"), "手链": ("服装鞋包", "配饰"), "耳环": ("服装鞋包", "配饰"),
    "戒指": ("服装鞋包", "配饰"), "围巾": ("服装鞋包", "配饰"), "手套": ("服装鞋包", "配饰"),
    # 手机数码
    "手机壳": ("手机数码", "手机配件"), "充电器": ("手机数码", "手机配件"), "数据线": ("手机数码", "手机配件"),
    "耳机": ("手机数码", "音频"), "音箱": ("手机数码", "音频"), "麦克风": ("手机数码", "音频"),
    "笔记本": ("手机数码", "电脑"), "显示器": ("手机数码", "电脑"), "键盘": ("手机数码", "电脑"),
    "鼠标": ("手机数码", "电脑"), "相机": ("手机数码", "相机"), "无人机": ("手机数码", "相机"),
    # 家电/家居
    "冰箱": ("家电", "大家电"), "洗衣机": ("家电", "大家电"), "电视": ("家电", "大家电"),
    "床单": ("家居生活", "家纺床品"), "被套": ("家居生活", "家纺床品"), "枕头": ("家居生活", "家纺床品"),
    "桌子": ("家居生活", "家具"), "椅子": ("家居生活", "家具"), "收纳": ("家居生活", "家具"),
    "锅": ("家居生活", "厨房用品"), "水杯": ("家居生活", "厨房用品"), "餐具": ("家居生活", "厨房用品"),
    # 母婴/运动
    "玩具": ("母婴玩具", "玩具"), "积木": ("母婴玩具", "玩具"), "娃娃": ("母婴玩具", "玩具"),
    "奶粉": ("母婴玩具", "尿裤喂养"), "纸尿裤": ("母婴玩具", "尿裤喂养"),
    "健身": ("运动户外", "健身器材"), "瑜伽": ("运动户外", "健身器材"), "自行车": ("运动户外", "骑行"),
    "帐篷": ("运动户外", "户外装备"), "足球": ("运动户外", "球类运动"), "羽毛球": ("运动户外", "球类运动"),
    # 美妆/健康
    "口红": ("美妆个护", "彩妆"), "面膜": ("美妆个护", "护肤"), "香水": ("美妆个护", "香水"),
    "洗发水": ("美妆个护", "洗护发"), "沐浴露": ("美妆个护", "身体护理"),
    "蛋白粉": ("健康保健", "营养保健"), "维生素": ("健康保健", "营养保健"), "按摩": ("健康保健", "个人护理"),
    # 食品
    "薯片": ("食品饮料", "零食"), "坚果": ("食品饮料", "零食"), "牛奶": ("食品饮料", "乳品蛋"),
    "啤酒": ("食品饮料", "酒类"), "矿泉水": ("食品饮料", "饮料"),
    # 宠物
    "猫粮": ("宠物用品", "宠物食品"), "狗粮": ("宠物用品", "宠物食品"),
    # 汽摩/爱好
    "摩托": ("汽车摩托配件", "摩托车"), "头盔": ("汽车摩托配件", "摩托车"),
    "手办": ("爱好收藏", "手办收藏"), "游戏": ("爱好收藏", "游戏"), "吉他": ("爱好收藏", "乐器"),
}


# 四、分类函数


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def classify_by_breadcrumb(path: List[str]) -> Optional[Tuple[str, str, str]]:
    """
    按中文 breadcrumb 路径分类
    Args:
        path: 中文类目路径列表（如 ["服装鞋包", "女装", "衬衫"]）
    Returns:
        (大类, 小类, 完整路径) 或 None（无法识别，交给关键词分类兜底）
    """
    if not path:
        return None
    # 大类：首段命中 CATEGORY_TREE 的一级类目名
    big = None
    first = (path[0] or "").strip()
    if first in CATEGORY_TREE:
        big = first
    else:
        for name in CATEGORY_TREE:
            if first and name in first:
                big = name
                break
    if big is None:
        return None
    # 小类：末段命中该大类下的中类/小类名
    small = None
    for seg in reversed(path):
        seg_text = (seg or "").strip()
        if not seg_text:
            continue
        for middle, smalls in CATEGORY_TREE[big].items():
            if seg_text == middle or seg_text in smalls:
                small = seg_text if seg_text in smalls else middle
                break
        if small:
            break
    if small is None:
        small = "其他"
    return big, small, f"{big}/{small}"


def classify_by_keywords(text: str) -> Tuple[str, str, str]:
    """
    按商品名关键词分类（在线 CSV 导入等无 breadcrumb 场景）。
    Returns: (大类, 小类, 完整路径)
    """
    if not text:
        return FALLBACK
    t = text.lower()
    # 1) 优先精确小类词
    for kw, (big, small) in SMALL_KEYWORD_MAP.items():
        if kw.lower() in t:
            return big, small, f"{big}/{small}"
    # 2) 大类规则
    for keywords, big, small in KEYWORD_MAP:
        if _contains_any(text, keywords):
            return big, small, f"{big}/{small}"
    return FALLBACK


def classify(text: str, breadcrumb: Optional[List[str]] = None) -> Tuple[str, str, str]:
    """统一入口：有 breadcrumb 优先用 breadcrumb，否则关键词分类。"""
    if breadcrumb:
        r = classify_by_breadcrumb(breadcrumb)
        if r:
            return r
    return classify_by_keywords(text)


# 五、类目树入库
def ensure_category_tree(cur) -> Dict[Tuple[str, str], int]:
    """
    把 CATEGORY_TREE 幂等写入 category 表（存在则跳过）。
    Returns: {(大类, 小类): category 表类目节点 id} 映射（用于商品关联类目树）
    """
    mapping: Dict[Tuple[str, str], int] = {}
    for big, middles in CATEGORY_TREE.items():
        # 大类
        cur.execute(
            "SELECT id FROM category WHERE name = %s AND level = 1",
            (big,),
        )
        row = cur.fetchone()
        if row:
            big_id = row[0]
        else:
            cur.execute(
                "INSERT INTO category (name, parent_id, level, path) VALUES (%s, NULL, 1, %s) RETURNING id",
                (big, big),
            )
            big_id = cur.fetchone()[0]

        for middle, smalls in middles.items():
            # 中类
            cur.execute(
                "SELECT id FROM category WHERE name = %s AND parent_id = %s",
                (middle, big_id),
            )
            row = cur.fetchone()
            if row:
                mid_id = row[0]
            else:
                mid_path = f"{big}/{middle}"
                cur.execute(
                    "INSERT INTO category (name, parent_id, level, path) VALUES (%s, %s, 2, %s) RETURNING id",
                    (middle, big_id, mid_path),
                )
                mid_id = cur.fetchone()[0]

            # 小类
            for small in smalls:
                cur.execute(
                    "SELECT id FROM category WHERE name = %s AND parent_id = %s",
                    (small, mid_id),
                )
                row = cur.fetchone()
                if row:
                    small_id = row[0]
                else:
                    small_path = f"{big}/{middle}/{small}"
                    cur.execute(
                        "INSERT INTO category (name, parent_id, level, path) VALUES (%s, %s, 3, %s) RETURNING id",
                        (small, mid_id, small_path),
                    )
                    small_id = cur.fetchone()[0]
                mapping[(big, small)] = small_id
    return mapping
