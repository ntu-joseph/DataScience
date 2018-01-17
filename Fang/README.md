
大家安安

TestTrain是訓練的PYTHON

TestPreproceccing是修改過後的PREPROCECCING

主要修改內容是增加權重的計算

原本的DURATION TIME 根據那周來乘一個權重 越晚權重越高(月*4+日/7)

還加了一個和原本DURATION的比值(權重版的有/15)

另外TIMESLOT也改成用權重的

dataTemp是訓練用資料
dataTest是測試用資料

目前這種弄法調了好幾次參數，最高80.~
