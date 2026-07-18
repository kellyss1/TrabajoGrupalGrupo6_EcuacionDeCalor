#include "palette.h"

/**
#FEE5D9
#FDD6C6
#FDC8B3
#FCB99F
#FCA98C
#FC9779
#FB8566
#FB7353
#F76245
#EF523C
#E84132
#E03128
#D32723
#C31F1E
#B4171A
#A50F15
*/

uint32_t bswap32(uint32_t a){
    return
    ((a & 0x000000FF) <<24)|
    ((a & 0x0000FF00) <<8)|
    ((a & 0x00FF0000) >>8)|
    ((a & 0xFF000000) >>24);
        
}

std::vector<uint32_t> color_ramp= {
bswap32(0XFF1010FF),
bswap32(0XF31017FF),
bswap32(0XE8101EFF),
bswap32(0XDC1126FF),
bswap32(0XD1112DFF),
bswap32(0XC51235FF),
bswap32(0XBA123CFF),
bswap32(0XAE1343FF),
bswap32(0XA3134BFF),
bswap32(0X971452FF),
bswap32(0X8C145AFF),
bswap32(0X801461FF),
bswap32(0X751568FF),
bswap32(0X691570FF),
bswap32(0X5E1677FF),
bswap32(0X52167FFF),
bswap32(0X471786FF),
bswap32(0X3B178DFF),
bswap32(0X301895FF),
bswap32(0X24189CFF),
bswap32(0X1919A4FF)


};

