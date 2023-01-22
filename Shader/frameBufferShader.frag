#version 450 core

layout (location = 0) out vec4 fragColor;
  
in vec2 Frag_UV;

uniform sampler2D screenTexture;

// ------------------------------------------ MOSAIC EFFECT -----------------------------------------------------
vec4 mosaicEffect()
{
    // TODO: insert real screen size instead of 493 and strength value instead of 3
    vec2 delta = clamp(vec2(3,3)/vec2(493,493),vec2(0),vec2(1));
    vec2 uv2 = fract(Frag_UV.st/delta);
    float len = 0.5 - length(uv2 - 0.5);

    float gamma=2.2;
    float exposure=1.;

    vec4 color = texture(screenTexture,Frag_UV.st);
    // HDR tonemapping
    color.rgb = vec3(1.0) - exp(-color.rgb*exposure);
    // gamma correction
    color.rgb = pow(color.rgb,vec3(1.0/gamma));

    return vec4(color.rgb,1.0) * smoothstep(0.0, 0.05, len);
}
// --------------------------------------------------------------------------------------------------------------

// ------------------------------------------ DISTORTION EFFECT -------------------------------------------------
vec3 mod289(vec3 x)
{
    return x-floor(x/289.)*289.;
}

vec4 mod289(vec4 x)
{
    return x-floor(x/289.)*289.;
}

vec4 permute(vec4 x)
{
    return mod289((x*34.+1.)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159-r*.85373472095314;
}

vec3 snoise_grad(vec3 v)
{
    const vec2 C=vec2(1./6.,1./3.);
    
    // First corner
    vec3 i=floor(v+dot(v,C.yyy));
    vec3 x0=v-i+dot(i,C.xxx);
    
    // Other corners
    vec3 g=step(x0.yzx,x0.xyz);
    vec3 l=1.-g;
    vec3 i1=min(g.xyz,l.zxy);
    vec3 i2=max(g.xyz,l.zxy);
    
    // x1 = x0 - i1  + 1.0 * C.xxx;
    // x2 = x0 - i2  + 2.0 * C.xxx;
    // x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1=x0-i1+C.xxx;
    vec3 x2=x0-i2+C.yyy;
    vec3 x3=x0-.5;
    
    // Permutations
    i=mod289(i);// Avoid truncation effects in permutation
    vec4 p=
    permute(permute(permute(i.z+vec4(0.,i1.z,i2.z,1.))
    +i.y+vec4(0.,i1.y,i2.y,1.))
    +i.x+vec4(0.,i1.x,i2.x,1.));
    
    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    vec4 j=p-49.*floor(p/49.);// mod(p,7*7)
    
    vec4 x_=floor(j/7.);
    vec4 y_=floor(j-7.*x_);// mod(j,N)
    
    vec4 x=(x_*2.+.5)/7.-1.;
    vec4 y=(y_*2.+.5)/7.-1.;
    
    vec4 h=1.-abs(x)-abs(y);
    
    vec4 b0=vec4(x.xy,y.xy);
    vec4 b1=vec4(x.zw,y.zw);
    
    //vec4 s0 = vec4(lessThan(b0, 0.0)) * 2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1, 0.0)) * 2.0 - 1.0;
    vec4 s0=floor(b0)*2.+1.;
    vec4 s1=floor(b1)*2.+1.;
    vec4 sh=-step(h,vec4(0.));
    
    vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
    vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
    
    vec3 g0=vec3(a0.xy,h.x);
    vec3 g1=vec3(a0.zw,h.y);
    vec3 g2=vec3(a1.xy,h.z);
    vec3 g3=vec3(a1.zw,h.w);
    
    // Normalise gradients
    vec4 norm=taylorInvSqrt(vec4(dot(g0,g0),dot(g1,g1),dot(g2,g2),dot(g3,g3)));
    g0*=norm.x;
    g1*=norm.y;
    g2*=norm.z;
    g3*=norm.w;
    
    // Compute gradient of noise function at P
    vec4 m=max(.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.);
    vec4 m2=m*m;
    vec4 m3=m2*m;
    vec4 m4=m2*m2;
    vec3 grad=
    -6.*m3.x*x0*dot(x0,g0)+m4.x*g0+
    -6.*m3.y*x1*dot(x1,g1)+m4.y*g1+
    -6.*m3.z*x2*dot(x2,g2)+m4.z*g2+
    -6.*m3.w*x3*dot(x3,g3)+m4.w*g3;
    return 42.*grad;
}

vec2 getRotationUV(vec2 uv, float angle, float power)
{
    vec2 v = vec2(0);
    float rad = angle * 3.1415926535;

    v.x = uv.x + cos(rad) * power;
    v.y = uv.y + sin(rad) * power;

    return v;
}

vec4 distortion()
{
    float distortionNoiseScale = 0.5;
    vec3 distortionNoisePos = vec3(0, 0, 0);
    float distortionPower = 1;

    vec3 uv1 = vec3(Frag_UV.st * distortionNoiseScale, 0);
    vec3 noise = snoise_grad(uv1 + distortionNoisePos);

    float gamma=2.2;
    float exposure=1.;

    vec2 uv = getRotationUV(Frag_UV.st, noise.x, noise.y * distortionPower);

    vec4 color=texture(screenTexture, uv.xy);
    // HDR tonemapping
    color.rgb=vec3(1.)-exp(-color.rgb*exposure);
    // gamma correction
    color.rgb=pow(color.rgb,vec3(1./gamma));
    
    return vec4(color.rgb,1.);
}
// ------------------------------------------------------------------------------------------------------------

// ----------------------------------------RGB SHIFT-----------------------------------------------------------
vec4 RGBShift()
{
    float gamma=2.2;
    float exposure=1.;
    vec2 _ShiftPower = vec2(2);

    vec2 shiftpow =_ShiftPower * (1./493);
    float r = texture(screenTexture, Frag_UV.st - shiftpow).r;
    vec2 ga = texture(screenTexture, Frag_UV.st).ga;
    float b = texture(screenTexture, Frag_UV.st + shiftpow).b;

    vec4 color = vec4(r, ga.x, b, ga.y);
    // HDR tonemapping
    color.rgb=vec3(1.)-exp(-color.rgb*exposure);
    // gamma correction
    color.rgb=pow(color.rgb,vec3(1./gamma));

    return color;
}
// ------------------------------------------------------------------------------------------------------------

// -----------------------------------RANDOM INVERT------------------------------------------------------------
float hash11(float p){
    vec2 p2=fract(p*vec2(443.8975,397.2973));
    p2+=dot(p2.xy,p2.yx+19.19);
    return fract(p2.x*p2.y);
}

float hash13(vec3 p){
    p=fract(p*vec3(443.8975,397.2973,491.1871));
    p+=dot(p.xyz,p.yzx+19.19);
    return fract(p.x*p.y*p.z);
}

vec2 hash21(float p){
    vec3 p3=fract(p*vec3(443.8975,397.2973,491.1871));
    p3+=dot(p3.xyz,p3.yzx+19.19);
    return fract(vec2(p3.x*p3.y,p3.z*p3.x));
}

float random(in float x){
    return fract(sin(x)*1e4);
}

float random(vec2 _st){
    return fract(sin(dot(_st.xy,vec2(12.9898,78.233)))*43758.5453123);
}

float random(vec3 _st){
    return fract(sin(dot(_st.xyz,vec3(12.9898,78.233,56.787)))*43758.5453123);
}

// Based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise(vec2 _st){
    vec2 i=floor(_st);
    vec2 f=fract(_st);
    
    // Four corners in 2D of a tile
    float a=random(i);
    float b=random(i+vec2(1.,0.));
    float c=random(i+vec2(0.,1.));
    float d=random(i+vec2(1.,1.));
    
    vec2 u=f*f*(3.-2.*f);
    
    return mix(a,b,u.x)+
    (c-a)*u.y*(1.-u.x)+
    (d-b)*u.x*u.y;
}

vec4 RandomInvert()
{
    float gamma=2.2;
    float exposure=1.;

    float _Threshold = 1;
    int _Invert = 0;
    float _StartTime = 0;
    float _Scale = 2;

    vec4 color=texture(screenTexture,Frag_UV.st);
    // HDR tonemapping
    color.rgb=vec3(1.)-exp(-color.rgb*exposure);
    // gamma correction
    color.rgb=pow(color.rgb,vec3(1./gamma));

    float r=noise(vec2(Frag_UV.st.y,_StartTime)*_Scale);

    color.rgb=(r>=_Threshold)?color.rgb:1-color.rgb;

    color.rgb=(_Invert==1)?1-color.rgb:color.rgb;

    return color;
}
// ------------------------------------------------------------------------------------------------------------

// ---------------------------------------------REFLECTION-----------------------------------------------------
vec4 Reflection()
{
    float gamma=2.2;
    float exposure=1.;
    int _Horizontal = 1;
    int _Vertical = 1;

    vec2 uv;
    uv.x=((_Horizontal == 1 && Frag_UV.st.x>.5)?1-Frag_UV.st.x:Frag_UV.st.x);
    uv.y=((_Vertical == 1 && Frag_UV.st.y>.5)?1-Frag_UV.st.y:Frag_UV.st.y);

    vec4 col=texture(screenTexture,uv);
    // HDR tonemapping
    col.rgb=vec3(1.)-exp(-col.rgb*exposure);
    // gamma correction
    col.rgb=pow(col.rgb,vec3(1./gamma));

    return col;
}
// ------------------------------------------------------------------------------------------------------------

// ---------------------------------------PIXELIZE-------------------------------------------------------------
vec4 Pixelize()
{
    float gamma = 2.2;
    float exposure = 1.0;
    
    float dx = 15.*(1./512.);
    float dy = 10.*(1./512.);
    vec2 coord = vec2(dx*floor(Frag_UV.x/dx),
                   dy*floor(Frag_UV.y/dy));
    vec4 color = texture(screenTexture, coord);
    // HDR tonemapping
    color.rgb = vec3(1.0) - exp(-color.rgb * exposure);
    // gamma correction
    color.rgb = pow(color.rgb, vec3(1.0 / gamma));

    return color;
}
// ------------------------------------------------------------------------------------------------------------

// ---------------------------------------------GAUSSIAN BLUR-----------------------------------------------------
vec4 gaussianBlur()
{
    float gamma = 2.2;
    float exposure = 1.0;
    float Pi = 6.28318530718; // Pi*2
    
    float Directions = 16.0; 
    float Quality = 3.0;
    float Size = 8.0;
   
    vec2 Radius = Size/ vec2(800, 600);

    vec4 color = texture(screenTexture, Frag_UV.st);
    // Blur calculations
    for( float d=0.0; d<Pi; d+=Pi/Directions)
    {
		for(float i=1.0/Quality; i<=1.0; i+=1.0/Quality)
        {
			color += texture( screenTexture, Frag_UV + vec2(cos(d),sin(d)) * Radius * i);		
        }
    } 
    // Output to screen
    color /= Quality * Directions - 15.0;
    color.rgb = vec3(1.0) - exp(-color.rgb * exposure);
    color.rgb = pow(color.rgb, vec3(1.0 / gamma));
    return color;
}
// ------------------------------------------------------------------------------------------------------------

// ---------------------------------------------MEDIAN FILTER + SOBEL EDGE DETECTION-----------------------------------------------------
mat3 sx = mat3( 
    1.0, 2.0, 1.0, 
    0.0, 0.0, 0.0, 
   -1.0, -2.0, -1.0 
);
mat3 sy = mat3( 
    1.0, 0.0, -1.0, 
    2.0, 0.0, -2.0, 
    1.0, 0.0, -1.0 
);

#define s2(a, b)				temp = a; a = min(a, b); b = max(temp, b);
#define mn3(a, b, c)			s2(a, b); s2(a, c);
#define mx3(a, b, c)			s2(b, c); s2(a, c);

#define mnmx3(a, b, c)			mx3(a, b, c); s2(a, b);                                   // 3 exchanges
#define mnmx4(a, b, c, d)		s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
#define mnmx5(a, b, c, d, e)	s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges

vec3 median() {

    float gamma = 2.2;
    float exposure = 1.0;

    vec3 v[9];
    mat3 I; 

    for(int dX = -1; dX <= 1; ++dX) {
        for(int dY = -1; dY <= 1; ++dY) {
            vec2 offset = vec2(float(dX), float(dY));
            vec3 c = vec3(texture2D(screenTexture, Frag_UV + offset * (1. / textureSize(screenTexture, 0))));
            c.rgb = vec3(1.0) - exp(-c.rgb * exposure);
            c.rgb = pow(c.rgb, vec3(1.0 / gamma));
            v[(dX + 1) * 3 + (dY + 1)] = c;
            I[dX + 1][dY + 1] = c.x * c.y * c.z;
        }
    }

    vec3 temp;
    vec3 orig = v[4];

    // Starting with a subset of size 6, remove the min and max each time
    mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
    mnmx5(v[1], v[2], v[3], v[4], v[6]);
    mnmx4(v[2], v[3], v[4], v[7]);
    mnmx3(v[3], v[4], v[8]);

    float gx = dot(sx[0], I[0]) + dot(sx[1], I[1]) + dot(sx[2], I[2]); 
    float gy = dot(sy[0], I[0]) + dot(sy[1], I[1]) + dot(sy[2], I[2]);

    float g = sqrt(pow(gx, 2.0)+pow(gy, 2.0));

    vec3 edgeColor = vec3(0.,0.,0.);
    return mix(v[4], edgeColor, g) - g;
}
// ------------------------------------------------------------------------------------------------------------

// ---------------------------------------------VIGNETTE-----------------------------------------------------
vec4 vignette(float falloff, float amount)
{
    float gamma = 2.2;
    float exposure = 1.0;
    vec4 color = texture(screenTexture, Frag_UV);

    float dist = distance(Frag_UV, vec2(0.5, 0.5));
    color.rgb *= smoothstep(0.8, falloff * 0.799, dist * (amount + falloff));

    color.rgb = vec3(1.0) - exp(-color.rgb * exposure);
    color.rgb = pow(color.rgb, vec3(1.0 / gamma));

    return color;
}
// ------------------------------------------------------------------------------------------------------------

// ------------------------------------------------OIL PAINTING-----------------------------------------------------------
// ------------------------------------------------STANDARD KUWAHARA FILTERING-----------------------------------------------------------

vec4 kuwahara()
{   
    vec4 fCol;
    float gamma = 2.2;
    float exposure = 1.0;
    int radius = 5;
    // Screen size
    vec2 scr_size = textureSize(screenTexture, 0);

    // Matrix subregions size
    float n = float((radius + 1) * (radius + 1));

    // Initialize the means and standard deviations to 0
    vec3 mean[4];
    vec3 s_dev[4];
    for(int i = 0; i < 4; ++i)
    {
        mean[i] = vec3(0.0);
        s_dev[i] = vec3(0.0);
    }

    // Get the subregions smaller and bigger coordinates
    struct SubRegion 
    {
        int x1, y1, x2, y2;
    };

    SubRegion S[4] = SubRegion[4](
        SubRegion(-radius, -radius, 0, 0),
        SubRegion(0, -radius, radius, 0),
        SubRegion(0, 0, radius, radius),
        SubRegion(-radius, 0, 0, radius)
    );
    
    // Compute the mean and standard deviation of each subregion
    for(int k = 0; k < 4; ++k)
    {
        for(int j = S[k].y1; j <= S[k].y2; ++j)
        {
            for(int i = S[k].x1; i <= S[k].x2; ++i)
            {
                vec3 c = texture(screenTexture, Frag_UV + vec2(i, j) / scr_size).rgb;
                // HDR tonemapping
                c = vec3(1.0) - exp(-c * exposure);
                // gamma correction
                c = pow(c, vec3(1.0 / gamma));
                mean[k] += c;
                s_dev[k] += c * c;
            }
        }
    }

    float min_sigma2 = 1e+1;
    for(int i = 0; i < 4; ++i)
    {
        // The last thing to do to compute the mean is to divide the value calculated before to the number of pixels in the subregion
        mean[i] /= n;

        // Similar thing is done for the standard deviation
        s_dev[i] = abs(s_dev[i] / n - mean[i] * mean[i]);

        float sigma2 = s_dev[i].r + s_dev[i].g + s_dev[i].b;
        // The output is the mean of a subregion with minimum variance
        if(sigma2 < min_sigma2)
        {
            min_sigma2 = sigma2;
            fCol = vec4(mean[i], 1.0);
        }
    }
    return fCol;
}
// ------------------------------------------------------------------------------------------------------------

void main()
{ 
    float gamma = 2.2;
    float exposure = 1.0;
    
    vec4 color = texture(screenTexture, Frag_UV.st);
    // HDR tonemapping
    color.rgb = vec3(1.0) - exp(-color.rgb * exposure);
    // gamma correction
    color.rgb = pow(color.rgb, vec3(1.0 / gamma));
    // f_objectID = ObjectID;
    fragColor = color.rgba;
   // fragColor = kuwahara();
}