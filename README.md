# XSformer

by *[Shuo Xu](https://www.researchgate.net/profile/Shuo-Xu-21).
****
🔔 This is the code of a X-ray Build-Up Factor Calculating Method for Multilayer Shields

🔔 We proposed a network based on [Transformer](https://arxiv.org/abs/1706.03762)
****

## background
Traditional method (empirical formula) to calculate Build-Up factor has large deviation
<center><img src="figures/e.bmp"		
                  alt="x"
                  height="300"/></center></td>
## Dataset
Te Monte Carlo method is uesd to generate dataset with different energy, different shield thickness, and different shield material combinations. 
Establishing concentric sphere model to improve simulation efficiency.

<table frame=void>
	<tr>		  
    <td><center><img src="figures/geo.png"		
                     alt="x"
                     height="250"/></center></td>	
    <td><center><img src="figures/geo2.png"		
                     alt="x"
                     height="200"/></center></td>	                     
    <td><center><img src="figures/mcnp.bmp"
                     alt="x"
                     height="200"/></center></td>
                     
  </tr>
</table>


## Network architecture
use Transformer as backbone.
<img src="figures/model.bmp">
