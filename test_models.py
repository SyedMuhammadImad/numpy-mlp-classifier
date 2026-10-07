import numpy as np
import pytest
import models as m

def test_full_mlp_gradient_against_finite_differences():
    np.random.seed(8)
    p = m.init_params([3,4,2]); x = np.random.normal(size=(5,3)); y = np.array([0,1,0,1,1])
    probs,cache = m.forward(x,p,training=False); grads=m.backward(probs,y,p,cache)
    for key,value in p.items():
        numerical=np.zeros_like(value)
        for index in np.ndindex(value.shape):
            old=value[index]; value[index]=old+1e-6
            plus=m.cross_entropy_loss(m.forward(x,p,training=False)[0],y)
            value[index]=old-1e-6
            minus=m.cross_entropy_loss(m.forward(x,p,training=False)[0],y)
            value[index]=old
            numerical[index]=(plus-minus)/2e-6
        np.testing.assert_allclose(grads['d'+key],numerical,atol=1e-7,rtol=1e-5)

def test_prediction_is_deterministic_and_softmax_is_stable():
    np.random.seed(9); p=m.init_params([3,8,2]);x=np.random.normal(size=(100,3))
    a=m.predict(x,p)
    for _ in range(10): np.testing.assert_array_equal(a,m.predict(x,p))
    probs=m.softmax(np.array([[10000.,10001.],[-10000.,-9999.]]))
    np.testing.assert_allclose(probs.sum(axis=1),1)
    np.testing.assert_allclose(probs[0],probs[1])
    with pytest.raises(ValueError):m.init_params([2,0,1])
    with pytest.raises(ValueError):m.forward(np.array([[np.nan]*3]),p)

def test_manual_training_learns_separable_classes():
    np.random.seed(11)
    x=np.r_[np.random.normal(-2,.2,(30,2)),np.random.normal(2,.2,(30,2))];y=np.r_[np.zeros(30,int),np.ones(30,int)]
    params,history=m.train(x,y,[2,8,2],epochs=80,lr=.05,verbose=False)
    assert np.mean(m.predict(x,params)==y)>.98
    assert history['loss'][-1]<history['loss'][0]
