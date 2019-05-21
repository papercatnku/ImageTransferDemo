# coding=utf-8
import mxnet as mx
import numpy as np
import cv2


from timeit import default_timer as timer

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
mx_mean=mx.nd.reshape(mx.nd.array(mean), (3, 1, 1))
mx_std = mx.nd.reshape(mx.nd.array(std), (3, 1, 1))

def npbgr2mxtsr(x, wh=None):
    if wh:
        x = cv2.resize(x, dsize=wh)
    mximg = mx.nd.array(x,dtype=np.uint8)
    imgtensor = mx.nd.image.to_tensor(mximg)
    imgtensor = mx.nd.image.normalize(imgtensor, mean=mean, std=std)
    return mx.nd.expand_dims(imgtensor, axis=0)

def mxtsr2nprgb(x):
    inv_norm_img = (x[0].as_in_context(mx.cpu()) * mx_std + mx_mean).asnumpy()
    inv_norm_img = np.transpose(
            255 * inv_norm_img, axes=(1, 2, 0)).clip(0, 255).astype(np.uint8)
    return inv_norm_img

class StyleTransfer(object):
    def __init__(self, gpuid=0, w=640, h=480):
        super(StyleTransfer,self).__init__()
        self.w = w
        self.h = h
        self.ctx = mx.gpu(gpuid) if gpuid >= 0 else mx.cpu()
        self.mod=None
        return
    
    def transfer(self, x_np):

        if self.mod is None:
            outimg = x_np
            return outimg, 1.0
            
        src_db= mx.io.DataBatch([npbgr2mxtsr(x_np).as_in_context(self.ctx)])
        inference_time = timer()
        try:
            self.mod.forward(src_db)
            stres = self.mod.get_outputs()[0]
            outimg = mxtsr2nprgb(stres)
        except:
            outimg = cv2.cvtColor(x_np,cv2.COLOR_BGR2RGB)
        inference_time = timer() - inference_time
        return outimg, inference_time

    def change_model(self, sym_fn, params_fn):
        try:
            self.mod = mx.mod.Module(
                symbol=mx.sym.load(sym_fn),
                data_names=('data',),
                label_names=None,
                context=self.ctx)
            self.mod.bind(
                data_shapes=[('data',(1, 3, self.h, self.w))],
                for_training=False,
                force_rebind=True)
            self.mod.load_params(params_fn)
        except:
            self.mod = None
        return


if __name__ == '__main__':
    import cv2
    srcsym_fn = './st-default.json'
    srcparams_fn = './st-default.params'
    srcimg_fn = './test.jpg'

    st = StyleTransfer(gpuid=-1,w=320,h=320)

    srcimg = cv2.cvtColor(cv2.imread(srcimg_fn), cv2.COLOR_BGR2RGB)

    res,t = st.transfer(srcimg)
    print('cost %.2f s'%t)

    cv2.imshow('res',cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    cv2.waitKey()

    ch_sym_fn = './models/weights/Picaso.json'
    ch_params_fn = './models/weights/Picaso.params'
    st.change_model(ch_sym_fn, ch_params_fn)

    res,t = st.transfer(srcimg)
    print('cost %.2f s'%t)

    cv2.imshow('res',cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    cv2.waitKey()

    cv2.destroyAllWindows()
