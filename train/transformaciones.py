import torch
#from torchvision import transforms


class Aumentador_Imagenes():
    def __init__(self, geometric_transforms, color_transforms_rgb,intensity_transforms,normalize_transforms):
        self.geometric_transforms = geometric_transforms
        self.color_transforms_rbg = color_transforms_rgb
        self.intensity_transforms=intensity_transforms
        self.normalize_transforms=normalize_transforms

    def __call__(self, img):
        ncanales=img.shape[0]
        #if ncanales is not 3:
         #   img1 = self.color_transforms(img[0:3]) # Solo a la imagen
          #  img1=torch.cat(img1,img[-1])
        #else:
        if self.color_transforms_rbg is not None:
            RGB=img[:3,:,:]
            RGB1 = self.color_transforms_rbg(RGB) # Solo a la imagen
            if img.shape[0] == 4 :
                canales_extra=img[3:,:,:]
                canales_extra1=self.intensity_transforms(canales_extra)
                img1=torch.concat([RGB1,canales_extra1],dim=0)
            elif img.shape[0] == 5 :
                canal_extra1=img[3:4,:,:]
                canal_extra1_transformed=self.intensity_transforms(canal_extra1)
                canal_extra2=img[4:5,:,:]
                canal_extra2_transformed=self.intensity_transforms(canal_extra2)           
                img1=torch.concat([RGB1,canal_extra1_transformed],dim=0)
                img1=torch.concat([img1,canal_extra2_transformed],dim=0)
            elif img.shape[0]==6:
                canal_extra=img[3:6,:,:]
                canal_extra_transformed=self.intensity_transforms(canal_extra)
                img1=torch.concat([RGB1,canal_extra_transformed],dim=0)
            elif img.shape[0] ==8 :
                canal_extra1=img[3:6,:,:]
                canal_extra1_transformed=self.intensity_transforms(canal_extra1)
                canal_extra2=img[6:7,:,:]
                canal_extra2_transformed=self.intensity_transforms(canal_extra2)
                canal_extra3=img[7:8,:,:]
                canal_extra3_transformed=self.intensity_transforms(canal_extra3)             
                img1=torch.concat([RGB1,canal_extra1_transformed],dim=0)
                img1=torch.concat([img1,canal_extra2_transformed],dim=0)
                img1=torch.concat([img1,canal_extra3_transformed],dim=0)
            else:
                img1=RGB1
        else:
            img1=img
        img1=self.normalize_transforms(img1)
        
        img2=self.geometric_transforms(img1)
        
        
            
        return img2