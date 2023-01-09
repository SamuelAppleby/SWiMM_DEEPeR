Shader "Nasty-Water/WaterShader"
{
	Properties
	{
		[HideInInspector]_FlowMap("Flow Map", 2D) = "white" {}
		[HideInInspector]_FlowSpeed("Flow Speed", Float) = 0.0
		_Normal("Normal", 2D) = "bump" {}
		_NormalScale("Normal Scale", Range(0,1)) = 0.1
			_Color("Color Blend", Color) = (1,1,1,1)
		_FColor("Fresnel Color", Color) = (1,1,1,1)
		_FScale("Fresnel Scale", Float) = 1.0
		_FPower("Fresnel Power", Float) = 1.0

		_WaveHeight("Wave Height", Range(0,10)) = 0.0
		_WaveSpeed("Wave Speed", Range(0,100)) = 1


			_DepthColor("Depth Color Blend", Color) = (1,1,1,1)
			_DepthThreshold("Depth Scale", Range(0,1.5)) = 0
			_Depth("Depth Height", Float) = 0
	}
		SubShader
	{
		Tags{ "Queue" = "Transparent"}
		LOD 100
			GrabPass
		{
			"_BackgroundTexture"
		}
		Pass
		{

			Cull Off ZWrite Off
			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"

			uniform sampler2D _CameraDepthTexture;

			struct appdata
			{
				float2 uv : TEXCOORD0;
				float4 vertex : POSITION;
				float3 normal : NORMAL;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 grabPos : TEXCOORD1;
				float4 projPos : TEXCOORD2;
				float3 worldDir : TEXCOORD3;
				UNITY_FOG_COORDS(1)
				float4 vertex : SV_POSITION;
				float R : TANGENT;
			};

			sampler2D _BackgroundTexture;
			sampler2D _Normal;
			sampler2D _FlowMap;
			float _NormalScale;

			fixed4 _FColor;
			fixed4 _Color;
			float _FScale;
			float _FPower;

			float _WaveHeight;
			float _WaveSpeed;

			float _DepthThreshold;
			float _Depth;
			float4 _DepthColor;
			float4x4 _ClipToWorld;
			float _FlowSpeed;

			v2f vert(appdata v)
			{
				v2f o;

				o.vertex = UnityObjectToClipPos(v.vertex);
				o.vertex.y += sin((_Time * _WaveSpeed) + v.vertex.x + v.vertex.z) * _WaveHeight;
				o.grabPos = ComputeGrabScreenPos(o.vertex);
				
				//depth

				o.projPos = ComputeScreenPos(o.vertex);
				float4 clip = float4(o.vertex.x, o.vertex.y,0.0,1.0); 
				o.worldDir = mul(_ClipToWorld, clip) - _WorldSpaceCameraPos;
				//o.worldDir = normalize(o.worldDir);
				//o.worldDir = _WorldSpaceCameraPos - float4(o.vertex.xy,0.0,1.0);
				//-
				
				o.uv = v.uv;

				float3 posWorld = mul(unity_ObjectToWorld, v.vertex).xyz;
				float3 normWorld = normalize(mul(unity_ObjectToWorld, v.normal));
				float3 viewDir = WorldSpaceViewDir(o.vertex);
				//o.worldDir = viewDir;

				//normWorld = refract(normalize(viewDir), normWorld, .0001);

				float3 I = normalize(posWorld - _WorldSpaceCameraPos.xyz);
				o.R = _FScale * pow(1.0 + dot(I, normWorld), _FPower);

				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				
				// depth


				float2 uv = i.projPos.xy / i.projPos.w;

				//float depth = Linear01Depth(tex2Dproj(_CameraDepthTexture,UNITY_PROJ_COORD(i.projPos)).r);

				float depth = LinearEyeDepth(tex2D(_CameraDepthTexture,uv).r);

				half dist = i.projPos.w - _Depth;
				depth = depth - dist;

				//float depth = tex2Dproj(_CameraDepthTexture,UNITY_PROJ_COORD(i.projPos));
				//depth = LinearEyeDepth(depth);

				//float3 dir = i.worldDir;
				//dir.z = depth;

				//float3 worldDepth = dir + _WorldSpaceCameraPos;
				float3 worldDepth = i.worldDir * depth + _WorldSpaceCameraPos.xyz;
				//worldDepth = worldDepth *_DepthThreshold;
				//worldDepth = normalize(worldDepth);
				//

				float2 flowMap = tex2D(_FlowMap, i.uv).rg * 2.0 - 1.0;

				half3 normal = UnpackNormal(tex2D(_Normal,i.uv + (flowMap+(_Time * _FlowSpeed)))) + sin((_Time * _WaveSpeed) + i.vertex.x + i.vertex.z) * _WaveHeight * 0.01;

				fixed4 col = tex2Dproj(_BackgroundTexture, UNITY_PROJ_COORD(i.grabPos + fixed4(normal * _NormalScale,0.0)));

				fixed4 fragCol = lerp(col * _Color, _FColor, i.R);
				
				fixed4 depthCol = lerp(fragCol, _DepthColor, saturate(depth * _DepthThreshold));

				fixed4 flowCol = lerp(depthCol, fixed4(1, 1, 1, 1), 1.0 - abs(flowMap.r * flowMap.g));

				return depthCol;
			}
			ENDCG
		}
	}
}
