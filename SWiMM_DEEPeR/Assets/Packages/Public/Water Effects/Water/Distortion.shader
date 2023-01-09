Shader "Nasty-Water/Distortion"
{
	Properties
	{
		[HideInInspector]_MainTex("Texture", 2D) = "white" {}
		_Normal("Normal", 2D) = "bump" {}

		_XScale("X Scale", Float) = 1.0
			_YScale("Y Scale", Float) = 1.0
			_Strength("Strength", Float) = 1.0

		_XSpeed("X Speed", Float) = 0.3
		_YSpeed("Y Speed", Float) = 0.3

		_Strength("Strength", Float) = 0.05

		_Color("Color Blend", Color) = (1,1,1,1)
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

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				return o;
			}

			sampler2D _MainTex;
			sampler2D _Normal;

			float _XScale;
			float _YScale;

			float _XSpeed;
			float _YSpeed;

			float _Strength;

			float4 _Color;

			fixed4 frag (v2f i) : SV_Target
			{
				fixed2 uv = i.uv;
				uv.x *= _XScale;
				uv.y *= _YScale;

				uv.x += _XSpeed * _Time;
				uv.y += _YSpeed * _Time;

				fixed3 normal = UnpackNormal(tex2D(_Normal, (uv) ));

				fixed4 col = tex2D(_MainTex, i.uv + (normal.xy * _Strength)) * _Color;
				return col;
			}
			ENDCG
		}
	}
}
