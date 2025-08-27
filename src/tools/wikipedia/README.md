# Wikipedia

## Japanese Wikipedia

1. [dumps wikimedia](https://dumps.wikimedia.org/other/enterprise_html/runs/)にアクセスする
2. 各言語のwikipedia dumpがあるので、日本語の場合は`jawiki`を、英語の場合は`enwiki`を選択する
3. `wget -c https://dumps.wikimedia.org/other/enterprise_html/runs/20250320/jawiki-NS0-20250320-ENTERPRISE-HTML.json.tar.gz`のように、ダウンロードを行う
4. `tar -xzvf jawiki-NS0-20250320-ENTERPRISE-HTML.json.tar.gz`で解凍する
5.
