"""
predict_end 变量用来标记预测是否完成。
初始化为 False，表示初始时预测还未完成,在预测完成后被设置为 True
以便其他部分的程序可以检查并做相应的处理。
"""
predict_end = False


"""
predic_Information_str 用来存储预测信息的字符串。
包含关于预测结果的文本描述，比如预测出的字符或其它相关的信息。
"""
predic_Information_str = ""


"""
predictchar
存储当前预测的字符， 初始化为 "a"。
在程序运行时，这个字符会根据模型的预测结果进行更新。
可以理解为是当前预测的“假设”值。
"""
predictchar = "a"


"""
sample_record_end 用来标记当前样本记录是否结束。
它指示是否已经完成一个样本的数据录入。
初始化为 False，表示样本记录还未结束。
"""
sample_record_end = False


"""
target_sign 记录目标字符的变量，初始化为 "a"。
用于标记当前正在记录的手势或目标字符。
"""
target_sign = "a"


"""
current_batch 表示当前批次的标记，初始化为 "0"。
当多次记录数据时，这个批次标识有助于区分不同的训练批次。
"""
current_batch = 0


"""
current_sample  用来记录当前样本的编号。初始化为 0，
随着每个样本的记录完成，这个编号会递增。
"""
current_sample = 0