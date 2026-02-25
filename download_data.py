import os
import requests
import time
from tqdm import tqdm
import re


def download_file(url, save_path):
    """
    下载文件
    
    Args:
        url: 文件URL
        save_path: 保存路径
    """
    print(f"正在下载: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def extract_wikipedia_text(dump_path, output_path):
    """
    提取维基百科文本
    
    Args:
        dump_path: 维基百科转储文件路径
        output_path: 输出文本路径
    """
    print("正在提取维基百科文本...")
    
    # 简单的XML解析
    import xml.etree.ElementTree as ET
    
    # 处理大型XML文件
    context = ET.iterparse(dump_path, events=('end',))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for event, elem in context:
            if elem.tag.endswith('page'):
                # 提取标题和文本
                title = ''
                text = ''
                
                for child in elem:
                    if child.tag.endswith('title'):
                        title = child.text
                    elif child.tag.endswith('revision'):
                        for rev_child in child:
                            if rev_child.tag.endswith('text'):
                                text = rev_child.text
                
                # 过滤非中文页面
                if title and text:
                    # 移除XML标签和模板
                    text = re.sub(r'\\{\\{.*?\\}\\}', '', text)
                    text = re.sub(r'\\[\\[.*?\\]\\]', '', text)
                    text = re.sub(r'<.*?>', '', text)
                    text = re.sub(r'\\n+', '\\n', text)
                    
                    # 只保留有意义的文本
                    if len(text) > 100:
                        f.write(f"{title}\n{text}\n\n")
                
                # 清理内存
                elem.clear()
    
    print(f"提取完成，保存到: {output_path}")


def download_chinese_wikipedia(output_path):
    """
    下载中文维基百科数据
    
    Args:
        output_path: 输出文件路径
    """
    # 中文维基百科最新转储地址
    url = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
    
    # 临时文件路径
    temp_file = "zhwiki-latest-pages-articles.xml.bz2"
    
    try:
        # 下载文件
        download_file(url, temp_file)
        
        # 解压文件
        print("正在解压文件...")
        import bz2
        with bz2.BZ2File(temp_file, 'rb') as fr, open('zhwiki-latest-pages-articles.xml', 'wb') as fw:
            for data in tqdm(fr, desc="解压中"):
                fw.write(data)
        
        # 提取文本
        extract_wikipedia_text('zhwiki-latest-pages-articles.xml', output_path)
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists('zhwiki-latest-pages-articles.xml'):
            os.remove('zhwiki-latest-pages-articles.xml')


def download_sample_corpus(output_path):
    """
    下载示例语料库
    
    Args:
        output_path: 输出文件路径
    """
    print("正在下载示例中文语料库...")
    
    # 使用CLUECorpus2020的小样本
    url = "https://github.com/CLUEbenchmark/CLUECorpus2020/raw/master/sample.txt"
    
    response = requests.get(url)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    print(f"示例语料库下载完成，保存到: {output_path}")


def main():
    """
    主函数
    """
    output_path = "e:\\CanFlyhang\\LLM\\data\\train.txt"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 下载示例语料库（快速）
    download_sample_corpus(output_path)
    
    # 如果你想要下载完整的维基百科数据，请取消下面的注释
    # 注意：这会下载几个GB的数据，需要较长时间
    # download_chinese_wikipedia(output_path)
    
    print("\n训练数据准备完成！")
    print(f"文件保存位置: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
