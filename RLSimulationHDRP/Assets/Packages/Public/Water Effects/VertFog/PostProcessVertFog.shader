Shader "Hidden/PostProcessVertFog"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
	_Threshold("Depth Scale", Float) = 0.01
		_Depth("Depth Height", Float) = 0.01
		_Color("Color", Color) = (1,1,1,1)
		_WaveSpeed("Wave Speed", Float) = 0.0
		_WaveHeight("Wave Height", Float) = 0.0
		_XFrequency("X frequency", Float) = 0.0
		_ZFrequency("Z frequency", Float) = 0.0
	}
	SubShader
	{
		Cull Off ZWrite Off ZTest Always

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			sampler2D _CameraDepthTexture;
			float4 _CameraDepthTexture_ST;

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float3 worldDirection : TEXCOORD1;
				float4 vertex : SV_POSITION;
			};

			float4x4 _ClipToWorld;

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				float4 clip = float4(o.vertex.xy, 0.0, 1.0);
				o.worldDirection = mul(_ClipToWorld, clip) - _WorldSpaceCameraPos;

				return o;
			}
			
			sampler2D _MainTex;
			float _Threshold;
			float _Depth;
			float _WaveHeight;
			float _WaveSpeed;
			float _ZFrequency;
			float _XFrequency;
			float4 _Color;

			fixed4 frag (v2f i) : SV_Target
			{
				float4 texel = tex2D(_MainTex,i.uv);

				float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv.xy);
			depth = LinearEyeDepth(depth);
			float3 worldspace = i.worldDirection * depth + _WorldSpaceCameraPos;


			worldspace.y += sin((_Time * _WaveSpeed) + (worldspace.x * _XFrequency)) * _WaveHeight;
			worldspace.y += sin((_Time * _WaveSpeed) + (worldspace.z * _ZFrequency)) * _WaveHeight;

			float4 color = lerp(_Color, texel,clamp((worldspace.y + _Depth) * _Threshold,0,1));
			return color;
			}
			ENDCG
		}
	}
}
