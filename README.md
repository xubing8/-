翻译记忆辅助工具
项目简介
翻译记忆辅助工具是一款基于 PyQt5 开发的桌面应用程序，旨在帮助翻译人员管理和使用翻译记忆库，提高翻译效率和一致性。该工具集成了多种翻译引擎，包括 百度翻译 API、Kimi API 和 OpenAI API，支持导入、导出和编辑翻译记忆库，并提供术语库管理功能。

功能特性
多引擎翻译支持：支持百度翻译 API、Kimi API 和 OpenAI API，可根据需求自由切换。
翻译记忆库管理：支持翻译记忆库的导入、导出和编辑，兼容 CSV、Excel、TMX 和 Word 文档格式。
术语库管理：自动提取关键术语，建立术语库，保证翻译的一致性和专业性。
相似句子匹配：利用 FAISS 向量检索技术，查找并应用翻译记忆库中的相似句子。
友好的用户界面：基于 PyQt5 的图形界面，操作简便直观，提升用户体验。
多语言支持：支持中文、英文、日文、韩文、法文、德文、俄文、西班牙文、印地语、阿拉伯语等多种语言的翻译。

安装指南
环境要求
Python 版本：3.6 及以上
操作系统：Windows、macOS 或 Linux
安装步骤
克隆项目


复制代码
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
创建虚拟环境（可选）

python -m venv venv
# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
安装依赖库
pip install \
    PyQt5 \
    transformers \
    torch \
    faiss-cpu \
    jieba \
    langdetect \
    nltk \
    pandas \
    openai \
    sentence-splitter \
    ratelimit \
    tenacity \
    anthropic \
    python-dotenv \
    LAC \
    python-docx
注意：确保您的环境中安装了 PyTorch，并根据您的硬件（CPU/GPU）选择合适的版本。

下载 NLTK 数据


import nltk
nltk.download('punkt')
配置指南
在使用应用程序之前，需要配置相关的 API 密钥。请在项目根目录下创建一个 .env 文件，并添加以下内容：


BAIDU_APP_ID=your_baidu_app_id
BAIDU_SECRET_KEY=your_baidu_secret_key
KIMI_API_KEY=your_kimi_api_key
KIMI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
ANTHROPIC_API_KEY=your_anthropic_api_key  # 可选
ANTHROPIC_BASE_URL=https://api.anthropic.com  # 可选

重要提示：

安全性：请确保不要将包含敏感信息的 .env 文件上传到公共仓库。
API 密钥获取：
百度翻译 API：前往 百度翻译开放平台 注册并获取。
Kimi API：根据官方指南获取相应的 API 密钥。
OpenAI API：前往 OpenAI 官方网站 注册并获取。
Anthropic Claude API：根据官方渠道获取（如果使用）。


**使用说明**

输入文本区域：在此输入需要翻译的文本。
源语言和目标语言选择：从下拉菜单中选择源语言和目标语言，支持自动检测。
翻译引擎选择：可选择百度翻译 API、Kimi API 或 OpenAI API。


**功能按钮：**
翻译：开始翻译输入的文本。
导入翻译记忆库：导入外部翻译记忆库文件，支持 CSV、Excel、TMX、Word 等格式。
导出翻译记忆库：将当前的翻译记忆库导出为指定格式。
查看记忆库：查看和编辑已有的翻译记忆库和术语库。
导入 Word 文档：导入需要翻译的 Word 文档内容。
导出为 Word：将翻译结果导出为 Word 文档。
清空记忆库：一键清除所有翻译记忆数据（操作不可撤销，请谨慎使用）。
翻译结果区域：显示源文本和译文的对照表。

**翻译流程**
输入文本：在输入区域输入需要翻译的文本，可以是段落或整篇文章。
设置语言和翻译引擎：选择源语言、目标语言和翻译引擎。
开始翻译：点击“翻译”按钮，应用程序将自动处理文本，包括分句、术语替换、查找相似句子等。
查看结果：翻译结果将在下方的输出区域以对照表形式显示。
管理翻译记忆库：可随时导入、导出或查看翻译记忆库，以保持翻译的一致性。

**功能详解**
多引擎翻译支持
百度翻译 API：需要设置 BAIDU_APP_ID 和 BAIDU_SECRET_KEY。
Kimi API：需要设置 KIMI_API_KEY，可自定义 KIMI_BASE_URL。
OpenAI API：需要设置 OPENAI_API_KEY，可自定义 OPENAI_BASE_URL。
引擎选择建议：根据实际需求和翻译质量选择合适的翻译引擎。
翻译记忆库管理
导入：支持导入 CSV、Excel、TMX、Word 等格式的翻译记忆库文件。
导出：可将翻译记忆库导出为上述格式，便于分享和备份。
编辑：内置记忆库编辑器，可对句子翻译、词汇翻译和术语库进行增删改查。
清空：提供一键清空记忆库的功能，注意此操作不可撤销。
术语库管理
自动提取：应用程序会自动从翻译文本中提取关键术语，并添加到术语库。
术语替换：在翻译过程中，优先使用术语库中的翻译，保证专业术语的一致性。
手动管理：可在记忆库查看界面手动添加、编辑或删除术语。
相似句子匹配
FAISS 索引：利用 FAISS 进行高效的相似句子检索。
向量化：使用 BERT 模型将句子向量化，计算相似度。
应用：在翻译时，若发现相似度高的句子，直接使用记忆库中的译文。


开发者指南
主要技术栈
PyQt5：构建图形用户界面。
Transformers：使用 BERT 模型进行句子向量化。
FAISS：高效的相似度检索工具。
Jieba：中文分词工具。
LAC：百度的分词与词性标注工具。
SQLite：本地数据库，存储翻译记忆数据。
NLTK：自然语言处理库，主要用于分句。
代码结构
translator_app.py：主应用程序代码，包含界面逻辑和主要功能实现。
translation_memory.db：SQLite 数据库文件，存储翻译记忆库和术语库。
faiss_index.bin：FAISS 索引文件，用于相似句子检索。
id_map.pkl：索引与数据库 ID 的映射文件。

常见问题
Q：运行程序时提示缺少依赖库。

A：请确保已按照安装指南正确安装所有依赖库，可使用 pip install -r requirements.txt 进行安装。

Q：翻译结果为空或出现错误信息。

A：请检查相应的 API 密钥是否正确配置，网络连接是否正常，以及目标语言是否支持。

Q：无法导入或导出翻译记忆库。

A：请确保文件格式正确，文件内容符合要求，必要时可查看日志获取详细错误信息。

贡献指南
欢迎对本项目提出建议、报告问题或提交 Pull Request。如果您有任何改进意见，欢迎与我们联系。

如何贡献
Fork 仓库：点击页面右上角的 Fork 按钮，创建项目的副本。



联系方式
如有任何问题或建议，请通过以下方式与我们联系：

邮箱：bisu520@88.com
GitHub Issues：欢迎在项目的 Issues 页面提出问题。



![image](https://github.com/user-attachments/assets/f2e92c90-799a-4275-b4ee-6dd5258c999e)

![image](https://github.com/user-attachments/assets/db82686b-aca5-4589-80cf-a6914a7ef3cf)


