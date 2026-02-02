一、英文分词 vs 中文分词
英文分词（Rule-based + 空格天然边界）
Rule-based: 靠人工规则切分文本，而不是学出来。例如遇到空格、标点就切。
英文有天然词边界：
I love natural language processing  ->  [I] [love] [natural] [language] [processing]
所以早期的NPL：token约等于word，分词用空格加标点即可。

中文分词（没有天然边界）
我爱自然语言处理 可以变成：  我 / 爱 / 自然 / 语言 / 处理；我 / 爱 / 自然语言 / 处理；我 / 爱 / 自然 / 语言处理
所以中文分词本身就是一个NPL序列标注任务。

Q为什么中文必须分词，英文不需要？
A因为英文存在显式的词边界（空格），而中文文本是连续字符串流，词边界时隐式的，需要通过统计或模型预测。


二、token / word / subword的区别
word 是语言学概念，token 是建模概念，subword 是工程折中方案
1.word（词）
人类语言单位，语义完整
word-level 建模，就是把“词（word）”作为模型能看到和处理的最小单位。这会导致OOV(out of vocabulary),输入中会出现词表里没有的词，只能统一映射成一个 <UNK>。随着新词越来越多，词表永远追不上真实世界。

2.token（模型真正处理的单位）
token是模型字典（vocab）中的最小可索引单位。token != word, token != 字。
token是一个工程概念。

3.subword（现代npl的核心）
subword 是 token 的一种实现方式，以下是几种分词算法
(1)BPE（Byte Pair Encoding）：不停把“最常一起出现的字符/片段”合并成一个 token。
核心思想：
        从字符级开始
        哪两个片段最常一起出现 → 合并
        一直合并到词表大小满足要求
BPE 是“贪心合并”

（2）WordPiece（BERT用的）：选择“最能提升语言模型概率”的 subword 切分方式。
因此BPE和WordPiece的区别是，前者只看重概率，后者更看重语言模型概率提升

（3）Unigram LM（SentencePiece用）：先给很多候选 subword，再“删掉没用的”。
思路与前两个相反：不是合并，是裁剪
核心思想：
        先准备一个很大的subword集合
        用概率模型评估
        不断删除贡献小的shbword

他们的本质都是通过subword建模，在控制词表规模的同时，显著降低OOV，并提升模型对未见词的泛化能力。


三、OOV（Out-of-Vocabulary）问题来源
本质：tokenizer 的词表无法覆盖输入字符串
三大根因：
        新词 / 热词，构建词表时不存在，如：ChatGPT、AIGC、元宇宙
        形态变化（英文），如果使用word-level模型，会导致同一个单词的不同时态全是新词，如：play → plays / played / playing
        拼写错误，或专有名词，如：xX_DarkKnight_666_Xx

Q为什么subword能缓解OOV
A因为subword使构建模型时的[词级]问题变成了[字符级]问题


四、为什么BERT不用[词]？
BERT不用word-level词表，是因为1、词表规模不可控 2、新词和形态变化导致严重的OOV 3、word-level无法泛化未见词，因此BERT使用subword tokenization（WordPiece），在控制词表规模的同时最大化覆盖率。


五、OOV例子
例1：
    输入：AIGC 技术发展迅速
    word-level：AIGC → OOV
    subword：A + I + G + C
例2：
    输入：He is unfriendliness
    word-level：unfriendliness → OOV
    subword：un + friend + li + ness

Q如果subword已经很好了，为什么还需要character-level？
A虽然主流大语言模型以 subword 为基本单位，但 subword 仍依赖固定词表，在面对拼写错误、噪声文本、极端新词或跨语言字符时仍存在不足；character-level 不存在 OOV，具有更强的鲁棒性，因此常作为补充或兜底方案，而不是主建模单位。