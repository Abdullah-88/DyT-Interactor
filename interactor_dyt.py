import torch
from torch import nn

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
    
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value      
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
       

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x
       
class GlobalDynamicTanh(nn.Module):
    def __init__(self, normalized_shape,sequence_length, alpha_init_value=0.5):
    
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value      
        self.alpha = nn.Parameter(torch.ones(normalized_shape*sequence_length) * alpha_init_value)
       

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x 

class MappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
           
        self.dyt_token = DynamicTanh(dim)
        self.gelu = nn.GELU()
        self.proj_1 =  nn.Linear(dim,dim,bias = False)
        self.proj_2 =  nn.Linear(dim,dim,bias = False)
        self.proj_3 = nn.Linear(dim,dim,bias = False)
             	   
    def forward(self, x):
    
    	x = self.dyt_token(x)    	
    	u, v = x, x 
    	u = self.proj_1(u)
    	u = self.gelu(u)
    	v = self.proj_2(v)
    	g = u * v
    	x = self.proj_3(g)
    	
    	
    	return x
    	

class InteractionUnit(nn.Module):
    def __init__(self,dim,num_tokens):
        super().__init__()
        
             
        self.dyt_token = DynamicTanh(dim) 
        self.dyt_context = GlobalDynamicTanh(dim,num_tokens)    
       
             	   
    def forward(self, x):
    
    	x = self.dyt_token(x)
    	dim0 = x.shape[0]
    	dim1 = x.shape[1]
    	dim2 = x.shape[2]
    	x = x.reshape([dim0,dim1*dim2])
    	
    	x = self.dyt_context(x)
    	
    	x = x.reshape([dim0,dim1,dim2])
    	
    	
    	
    	return x
    	

class InteractorBlock(nn.Module):
    def __init__(self, d_model, num_tokens):
        super().__init__()
       
         
        self.mapping = MappingUnit(d_model)
        self.interaction = InteractionUnit(d_model,num_tokens)
        
    def forward(self, x):
                  
        residual = x
        
        x = self.interaction(x)
    
        x = x + residual
        
        residual = x
        
        x = self.mapping(x)
        
                                          
        out = x + residual
        
        
        return out



class Interactor(nn.Module):
    def __init__(self, d_model,num_tokens, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[InteractorBlock(d_model,num_tokens) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








