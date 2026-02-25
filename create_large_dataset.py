import os


def create_large_chinese_corpus(output_path):
    """
    创建大型中文语料库
    
    Args:
        output_path: 输出文件路径
    """
    print("正在创建大型中文语料库...")
    
    # 中文语料库
    corpus = []
    
    # 1. 中文新闻文本（100条）
    print("添加中文新闻文本...")
    news_categories = [
        "政治", "经济", "科技", "文化", "体育",
        "教育", "医疗", "环保", "交通", "国际"
    ]
    
    for category in news_categories:
        for i in range(10):
            news = f"{category}新闻：据新华社报道，近日{category}领域发生了重要事件。相关部门表示，将采取一系列措施促进{category}事业的发展。专家认为，这一举措将对{category}领域产生深远影响，有利于推动相关产业的升级和转型。"
            corpus.append(news)
    
    # 2. 中文故事文本（50条）
    print("添加中文故事文本...")
    story_themes = [
        "友情", "亲情", "爱情", "勇气", "智慧",
        "诚信", "善良", "坚持", "梦想", "希望"
    ]
    
    for theme in story_themes:
        for i in range(5):
            story = f"关于{theme}的故事：从前有一个年轻人，他一直相信{theme}的力量。在面对困难时，他始终保持着{theme}的信念，最终克服了所有挑战，实现了自己的目标。这个故事告诉我们，{theme}是人生中最宝贵的品质之一。"
            corpus.append(story)
    
    # 3. 中文科普文本（50条）
    print("添加中文科普文本...")
    science_topics = [
        "天文学", "地理学", "生物学", "化学", "物理学",
        "数学", "计算机科学", "医学", "心理学", "环境科学"
    ]
    
    for topic in science_topics:
        for i in range(5):
            science = f"{topic}科普：{topic}是一门研究自然现象和规律的科学。通过{topic}的研究，我们可以更好地理解世界的运行机制，开发出更多有利于人类的技术和产品。{topic}的发展离不开科学家们的不懈努力和创新精神。"
            corpus.append(science)
    
    # 4. 中文历史文本（50条）
    print("添加中文历史文本...")
    dynasties = [
        "夏朝", "商朝", "周朝", "秦朝", "汉朝",
        "三国", "晋朝", "隋朝", "唐朝", "宋朝"
    ]
    
    for dynasty in dynasties:
        for i in range(5):
            history = f"{dynasty}历史：{dynasty}是中国历史上的一个重要朝代，存在于公元前{1000+i*100}年至公元前{900+i*100}年左右。{dynasty}时期，政治稳定，经济繁荣，文化昌盛，出现了许多杰出的政治家、军事家、文学家和科学家。"
            corpus.append(history)
    
    # 5. 中文文学文本（50条）
    print("添加中文文学文本...")
    literary_forms = [
        "诗歌", "散文", "小说", "戏剧", "寓言",
        "童话", "神话", "传说", "随笔", "评论"
    ]
    
    for form in literary_forms:
        for i in range(5):
            literature = f"{form}欣赏：{form}是一种优美的文学形式，通过文字的艺术表达，传递作者的情感和思想。优秀的{form}作品能够打动人心，引发读者的共鸣，成为文化传承的重要载体。"
            corpus.append(literature)
    
    # 6. 中文哲学文本（50条）
    print("添加中文哲学文本...")
    philosophical_schools = [
        "儒家", "道家", "法家", "墨家", "名家",
        "阴阳家", "纵横家", "杂家", "农家", "小说家"
    ]
    
    for school in philosophical_schools:
        for i in range(5):
            philosophy = f"{school}思想：{school}是中国古代重要的哲学流派之一，代表人物有{school}的创始人及其弟子。{school}的核心思想包括对人与自然、人与社会、人与人关系的探讨，对中国传统文化产生了深远的影响。"
            corpus.append(philosophy)
    
    # 7. 中文谚语和成语故事（50条）
    print("添加中文谚语和成语故事...")
    idioms = [
        "井底之蛙", "守株待兔", "拔苗助长", "掩耳盗铃", "亡羊补牢",
        "画蛇添足", "叶公好龙", "狐假虎威", "囫囵吞枣", "井底之蛙",
        "刻舟求剑", "滥竽充数", "买椟还珠", "盲人摸象", "南辕北辙",
        "杞人忧天", "黔驴技穷", "杀鸡取卵", "守株待兔", "水落石出",
        "铁杵磨针", "亡羊补牢", "望梅止渴", "胸有成竹", "雪中送炭",
        "掩耳盗铃", "叶公好龙", "夜郎自大", "愚公移山", "鹬蚌相争",
        "缘木求鱼", "凿壁偷光", "自相矛盾", "八仙过海", "百步穿杨",
        "班门弄斧", "杯弓蛇影", "不耻下问", "才高八斗", "草木皆兵",
        "车水马龙", "乘风破浪", "出类拔萃", "唇亡齿寒", "打草惊蛇",
        "大器晚成", "大义凛然", "呆若木鸡", "道听途说", "得陇望蜀"
    ]
    
    for idiom in idioms:
        idiom_story = f"{idiom}：{idiom}是一个常用的成语，用来形容{idiom}的意思。这个成语来源于古代的一个故事，讲述了{idiom}的典故。现在，人们常用{idiom}来比喻某种情况或行为。"
        corpus.append(idiom_story)
    
    # 8. 中文日常对话（50条）
    print("添加中文日常对话...")
    conversation_scenarios = [
        "问候", "介绍", "感谢", "道歉", "邀请",
        "告别", "问路", "购物", "就餐", "就医"
    ]
    
    for scenario in conversation_scenarios:
        for i in range(5):
            conversation = f"{scenario}对话：在{scenario}的场景中，人们通常会说一些特定的话语。例如，当{scenario}时，可以说'你好，很高兴见到你'，'请问有什么可以帮助你的吗'等。这些对话是日常生活中不可或缺的一部分。"
            corpus.append(conversation)
    
    # 9. 中文教育文本（50条）
    print("添加中文教育文本...")
    education_levels = [
        "学前教育", "小学教育", "中学教育", "大学教育", "研究生教育",
        "职业教育", "成人教育", "远程教育", "特殊教育", "终身教育"
    ]
    
    for level in education_levels:
        for i in range(5):
            education = f"{level}：{level}是教育体系中的重要组成部分，针对不同年龄段和需求的学习者。{level}的目标是培养学生的知识、技能和价值观，为他们的未来发展奠定基础。"
            corpus.append(education)
    
    # 10. 中文商业文本（50条）
    print("添加中文商业文本...")
    business_areas = [
        "市场营销", "人力资源", "财务管理", "战略规划", "运营管理",
        "客户服务", "供应链管理", "产品开发", "品牌建设", "创业创新"
    ]
    
    for area in business_areas:
        for i in range(5):
            business = f"{area}：{area}是企业管理中的重要领域，涉及到企业的各个方面。有效的{area}策略可以帮助企业提高效率，降低成本，增强竞争力，实现可持续发展。"
            corpus.append(business)
    
    # 11. 中文科技文本（50条）
    print("添加中文科技文本...")
    tech_fields = [
        "人工智能", "大数据", "云计算", "物联网", "区块链",
        "虚拟现实", "增强现实", "量子计算", "5G技术", "生物技术"
    ]
    
    for field in tech_fields:
        for i in range(5):
            tech = f"{field}：{field}是当今科技领域的热门话题，正在改变我们的生活和工作方式。{field}的发展速度非常快，不断涌现出新的应用和解决方案，为人类社会的进步做出了重要贡献。"
            corpus.append(tech)
    
    # 12. 中文健康文本（50条）
    print("添加中文健康文本...")
    health_topics = [
        "饮食健康", "运动健康", "心理健康", "睡眠健康", "预防疾病",
        "慢性病管理", "急救知识", "健康体检", "合理用药", "健康生活方式"
    ]
    
    for topic in health_topics:
        for i in range(5):
            health = f"{topic}：{topic}是保持身体健康的重要因素。专家建议，我们应该注重{topic}，养成良好的生活习惯，定期进行健康检查，及时发现和处理健康问题。"
            corpus.append(health)
    
    # 13. 中文旅游文本（50条）
    print("添加中文旅游文本...")
    tourist_attractions = [
        "自然风光", "历史古迹", "文化遗产", "城市景观", "乡村旅游",
        "生态旅游", "红色旅游", "主题公园", "博物馆", "美食之旅"
    ]
    
    for attraction in tourist_attractions:
        for i in range(5):
            tourism = f"{attraction}：{attraction}是旅游的重要组成部分，吸引着大量的游客前来参观和体验。{attraction}不仅可以让人们放松身心，还可以增长见识，了解不同地区的文化和历史。"
            corpus.append(tourism)
    
    # 14. 中文艺术文本（50条）
    print("添加中文艺术文本...")
    art_forms = [
        "绘画", "书法", "雕塑", "音乐", "舞蹈",
        "戏剧", "电影", "摄影", "设计", "民间艺术"
    ]
    
    for form in art_forms:
        for i in range(5):
            art = f"{form}艺术：{form}是一种创造性的艺术形式，通过独特的表达方式传递艺术家的情感和思想。优秀的{form}作品能够打动人心，成为文化传承的重要载体。"
            corpus.append(art)
    
    # 15. 中文体育文本（50条）
    print("添加中文体育文本...")
    sports = [
        "足球", "篮球", "排球", "乒乓球", "羽毛球",
        "网球", "游泳", "跑步", "健身", "武术"
    ]
    
    for sport in sports:
        for i in range(5):
            sports_text = f"{sport}运动：{sport}是一项受欢迎的体育运动，不仅可以锻炼身体，还可以培养团队合作精神和竞争意识。{sport}运动有着悠久的历史和广泛的群众基础，是全民健身的重要组成部分。"
            corpus.append(sports_text)
    
    # 保存到文件
    print("保存中文语料库...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in corpus:
            f.write(text + '\n\n')
    
    print(f"\n大型中文语料库创建完成！")
    print(f"文件保存位置: {output_path}")
    print(f"文本数量: {len(corpus)}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    """
    主函数
    """
    output_path = "e:\\CanFlyhang\\LLM\\data\\train.txt"
    create_large_chinese_corpus(output_path)


if __name__ == "__main__":
    main()
