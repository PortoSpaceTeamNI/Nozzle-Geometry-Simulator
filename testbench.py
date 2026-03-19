from CoolProp.CoolProp import PropsSI
Ttank = 298.15 #K (Temp ambiente)
Ptank = 300e5 #Pa (300 bar exemplo)

rho = PropsSI('D', 'T', Ttank, 'P', Ptank, 'Nitrogen')
print(rho)