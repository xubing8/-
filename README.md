项目简介
翻译记忆辅助工具是一款基于 PyQt5 开发的桌面应用程序，旨在帮助翻译人员管理和使用翻译记忆库，提高翻译效率和一致性。该工具集成了多种翻译引擎，包括百度翻译 API、Kimi API 和 OpenAI API，支持导入、导出和编辑翻译记忆库，并提供术语库管理功能。

功能特性
多引擎翻译支持：可选择百度翻译 API、Kimi API 和 OpenAI API 进行翻译。
翻译记忆库管理：支持翻译记忆库的导入、导出和编辑，兼容 CSV、Excel、TMX 和 Word 文档格式。
术语库管理：自动提取关键术语，建立术语库，保证翻译的一致性。
相似句子匹配：利用 FAISS 向量检索技术，查找并应用翻译记忆库中的相似句子。
友好的用户界面：基于 PyQt5 的 GUI 界面，操作简便直观。
安装指南
环境要求
Python 3.6 及以上版本
操作系统：Windows、macOS 或 Linux
安装步骤
克隆项目

bash
复制代码
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
创建虚拟环境（可选）

bash
复制代码
python -m venv venv
# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
安装依赖库

bash
复制代码
pip install -r requirements.txt
如果没有 requirements.txt 文件，请手动安装所需的依赖库：

bash
复制代码
pip install PyQt5 transformers torch faiss-cpu jieba langdetect nltk pandas openai sentence-splitter ratelimit tenacity anthropic python-dotenv LAC python-docx
使用说明
配置 API 密钥

百度翻译 API：前往 百度翻译开放平台 注册并获取 APP_ID 和 SECRET_KEY。
Kimi API：获取 KIMI_API_KEY 并设置 KIMI_BASE_URL（如果不同于默认值）。
OpenAI API：获取 OPENAI_API_KEY 并设置 OPENAI_BASE_URL（如果使用代理服务）。
Anthropic Claude API（可选）：获取 ANTHROPIC_API_KEY 并设置 ANTHROPIC_BASE_URL。
将上述密钥和配置添加到项目根目录下的 .env 文件中：

env
复制代码
BAIDU_APP_ID=your_baidu_app_id
BAIDU_SECRET_KEY=your_baidu_secret_key
KIMI_API_KEY=your_kimi_api_key
KIMI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_BASE_URL=https://api.anthropic.com
运行应用程序

bash
复制代码
python translator_app.py
使用界面

输入文本：在输入框中输入需要翻译的文本。
选择语言：设置源语言和目标语言，支持自动检测。
选择翻译引擎：从下拉菜单中选择所需的翻译引擎。
翻译：点击“翻译”按钮获取翻译结果。
管理翻译记忆库：
导入记忆库：支持 CSV、Excel、TMX、Word 等格式。
导出记忆库：将当前的翻译记忆库导出为所需格式。
查看记忆库：查看和编辑已有的翻译记忆库和术语库。
清空记忆库：一键清除所有翻译记忆数据。
导入/导出 Word 文档：直接导入需要翻译的 Word 文档，或将翻译结果导出为 Word 文件。
注意事项
API 密钥安全：请确保您的 API 密钥安全，避免在公共场合泄露。建议使用 .env 文件或环境变量来存储密钥。
依赖库版本：某些依赖库可能需要特定版本，建议根据需要调整 requirements.txt 或手动安装指定版本的库。
翻译质量：不同的翻译引擎可能有不同的翻译效果，可根据实际需求选择合适的引擎。
贡献
欢迎对本项目提出建议、报告问题或提交拉取请求。如果您有任何改进意见，欢迎与我们联系。

许可证
本项目采用 MIT 许可证进行许可。详情请参阅 LICENSE 文件。


![image](https://github.com/user-attachments/assets/f2e92c90-799a-4275-b4ee-6dd5258c999e)



