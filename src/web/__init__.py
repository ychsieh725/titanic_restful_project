"""展示層（Presentation）。

此 package 為 Flask glue：把 HTTP request/response 翻譯為 src.ml 服務層的
函式呼叫。可 import Flask 與 src.ml；服務層（src/ml/）反之不得依賴本層，
以維持框架解耦（CON-4）。
"""
