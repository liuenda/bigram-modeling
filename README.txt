2016/7/18
phrase_det.py 作为一个module存在不单独运行
phrase_det将会在训练word2vec模型的之前，自动检测文章中的phrases，然后重新形成一个一组corpus来做word2vec训练
运行顺序
. training.py 
	. 有参数，默认为训练EN，也可以同时训练日语和英语
	. 加载了phrase_det模块，在运行的时候自动识别词组
	. 通过phrase_flag=1（默认一直等于1）来判断是否启用词组识别模式
. training-jp.py(可选)