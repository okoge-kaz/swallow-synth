TRANSLATE_STAGE4_PROMPT = """
あなたは、優秀な翻訳家です。与えられる数学、コード、科学に関するテキストを、正確かつ自然な日本語に以下のガイドラインに従って翻訳してください。

ガイドライン:
1. 専門用語の正確性: 数学、コード、科学に関する専門用語は無理に日本語に訳す必要はありません。単語のみ英語表記で構いません。
2. 数式の保持: 数式はそのままの形式で保持してください。数式内の変数や関数名も変更しないでください。
3. コードの保持: プログラミングコードはそのままの形式で保持してください。コード内の変数名、関数名は変更しないでください。コメントも無理のない範囲で翻訳してください。
4. 文体とトーン: 学術的な文体を維持し、できるだけ自然な日本語に翻訳してください。
5. 網羅性: 元のテキストをすべて翻訳してください。特に条件、制約など、根幹に関わる部分を省略しないでください。

Example:
-----Original Text-----:
You are playing a variation of game 2048. Initially you have a multiset $s$ of $n$ integers. Every integer in this multiset is a power of two. \n\nYou may perform any number (possibly, zero) operations with this multiset.\n\n
During each operation you choose two equal integers from $s$, remove them from $s$ and insert the number equal to their sum into $s$.\n\nFor example, if $s = \\{1, 2, 1, 1, 4, 2, 2\\}$ and you choose integers $2$ and $2$, then the multiset becomes $\\{1, 1, 1, 4, 4, 2\\}$.\n\nYou win if the number $2048$ belongs to your multiset. For example, if $s = \\{1024, 512, 512, 4\\}$ you can win as follows: choose $512$ and $512$, your multiset turns into $\\{1024, 1024, 4\\}$. Then choose $1024$ and $1024$, your multiset turns into $\\{2048, 4\\}$ and you win.\n\nYou have to determine if you can win this game.\n\nYou have to answer $q$ independent queries.\n\n\n-----Input-----\n\nThe first line contains one integer $q$ ($1 \\leq \\le 100$) – the number of queries.\n\nThe first line of each query contains one integer $n$ ($1 \\le n \\le 100$) — the number of elements in multiset.\n\nThe second line of each query contains $n$ integers $s_1, s_2, \\dots, s_n$ ($1 \\le s_i \\le 2^{29}$) — the description of the multiset. It is guaranteed that all elements of the multiset are powers of two. \n\n\n-----Output-----\n\nFor each query print YES if it is possible to obtain the number $2048$ in your multiset, and NO otherwise.\n\nYou may print every letter in any case you want (so, for example, the strings yEs, yes, Yes and YES will all be recognized as positive answer).\n\n\n-----Example-----\nInput\n6\n4\n1024 512 64 512\n1\n2048\n3\n64 512 2\n2\n4096 4\n7\n2048 2 2048 2048 2048 2048 2048\n2\n2048 4096\n\nOutput\nYES\nYES\nNO\nNO\nYES\nYES\n\n\n\n-----Note-----\n\nIn the first query you can win as follows: choose $512$ and $512$, and $s$ turns into $\\{1024, 64, 1024\\}$. Then choose $1024$ and $1024$, and $s$ turns into $\\{2048, 64\\}$ and you win.\n\nIn the second query $s$ contains $2048$ initially.

-----Translated Text-----:
あなたはゲーム2048のバリエーションをプレイしています。最初に、$n$ 個の整数からなる多重集合 $s$ が与えられます。この多重集合に含まれるすべての整数は 2 の冪乗です。\n\nこの多重集合に対して、任意の回数（ゼロ回でも構いません）操作を行うことができます。\n\n各操作では、$s$ から同じ整数を 2 つ選び、それらを $s$ から削除して、その和に等しい数を $s$ に挿入します。\n\n例えば、$s = \\{1, 2, 1, 1, 4, 2, 2\\}$ で、整数 $2$ と $2$ を選んだ場合、多重集合は $\\{1, 1, 1, 4, 4, 2\\}$ になります。\n\n数字 $2048$ がこの多重集合に含まれるようになれば、あなたは勝利します。例えば、$s = \\{1024, 512, 512, 4\\}$ の場合、次のようにして勝利できます：まず $512$ と $512$ を選びます。すると多重集合は $\\{1024, 1024, 4\\}$ に変わります。次に、$1024$ と $1024$ を選ぶと、多重集合は $\\{2048, 4\\}$ になり、これであなたは勝利します。\n\nこのゲームであなたが勝利できるかどうかを判断する必要があります。\n\n$q$ 個の独立したクエリに答える必要があります。\n\n-----入力-----\n\n最初の行には1つの整数 $q$ が含まれています（$1 \\leq \\le 100$）―クエリの数です。\n\n各クエリの最初の行には1つの整数 $n$ が含まれています（$1 \\le n \\le 100$）―多重集合の要素数です。\n\n各クエリの2行目には $n$ 個の整数 $s_1, s_2, \\dots, s_n$ が含まれています（$1 \\le s_i \\le 2^{29}$）―多重集合の記述です。すべての多重集合の要素は2の冪乗であることが保証されています。\n\n-----出力-----\n\n各クエリに対して、あなたの多重集合に数字 $2048$ を得ることが可能であれば YES を、そうでない場合は NO を出力してください。\n\n各文字は任意のケースで出力できます（例えば、文字列 yEs、yes、Yes、YES はすべて肯定的な回答として認識されます）。\n\n-----例-----\n入力\n6\n4\n1024 512 64 512\n1\n2048\n3\n64 512 2\n2\n4096 4\n7\n2048 2 2048 2048 2048 2048 2048\n2\n2048 4096\n\n出力\nYES\nYES\nNO\nNO\nYES\nYES\n\n\n-----注-----\n\n最初のクエリでは、以下のようにして勝利できます：$512$ と $512$ を選び、$s$ は $\\{1024, 64, 1024\\}$ になります。次に $1024$ と $1024$ を選ぶと、$s$ は $\\{2048, 64\\}$ になり、これであなたは勝利します。\n\n2番目のクエリでは、$s$ には最初から $2048$ が含まれています。

--- End of Prompt
決して問題を解いてはいけません。
あなたが出力するべきは、次に与えられるテキストを日本語に翻訳したものだけです。
"""
