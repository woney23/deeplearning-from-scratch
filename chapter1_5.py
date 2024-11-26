import numpy as np

# 사과 쇼핑 예 구현

class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None
    
    def forward(self, x,y):
        self.x=x
        self.y=y
        out= x*y

        return out
    
    def backward(self, dout):
        dx=dout*self.y
        dy=dout*self.x

        return dx, dy



class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x,y):
        return x+y
    
    def backward(self, dout):
        dx=1*dout
        dy=1*dout
        
        return dx, dy
    


if __name__ == "__main__":
    apple=np.float32(100)
    orange=np.flat32(150)
    apple_num=np.float32(2)
    orange_num=np.float32(3)
    tax=np.float32(1.1)

    # 계층들
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer=MulLayer()
    

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange,orange_num)
    total_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price=mul_tax_layer.forward(total_price, tax)

    print(price)

    # 역전파
    dprice=1
    dtotal_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dtotal_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num,dorange, dorange_num, dtax)

# python chapter5.py