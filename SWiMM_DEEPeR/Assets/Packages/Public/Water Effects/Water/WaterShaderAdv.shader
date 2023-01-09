
Shader "Nasty-Water/WaterShaderAdv" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_MainTex("Normal", 2D) = "bump" {}
		_SecondaryTex("Secondary Normal", 2D) = "bump" {}
		_NoiseMap("Disruption Map", 2D) = "white" {}
		[Space(25)]
		_FlowMap("Flow Map", 2D) = "white" {}
		_FlowSpeed("Flow Speed", Float) = 1.0
		_FlowCycleLength("Flow Cycle", Float) = 1.0
		[Space(25)]
		[Toggle]_ToonShaded ("Is Toon Shaded", Float) = 0
		_Glossiness ("Specular", Float) = 0.5
		_Metallic ("Specular Power", Float) = 0.0
		[Space(25)]
		_WaveSpeed("Wave Speed" , Float) = 0.0
		_WaveHeight("Wave Height", Float) = 0.0
		_XFreq("X Frequency", Float) = 0.0
		_ZFreq("Z Frequency", Float) = 0.0
	}
		SubShader{
			Tags{ "Queue" = "Geometry+1" "RenderType" = "Transparent"}

			GrabPass{ "_BackgroundTexture" }

			CGPROGRAM
#pragma surface surf CelShading vertex:vert fullforwardshadows
#pragma debug
#pragma target 3.0

		bool _ToonShaded;

		inline half4 LightingCelShading(SurfaceOutput s, half3 lightDir, half3 viewDir, half atten) {
			//diffuse
			half NdotL = dot(s.Normal, lightDir);

			if (_ToonShaded)
			{
				if (NdotL <= 0.0)NdotL = 0.0;
				else NdotL = 1.0;
			}
			else 
			{
				NdotL = clamp(NdotL, 0, 1);
			}

			half TAtten = atten;
			if (_ToonShaded)
			{
				if (TAtten > .0) TAtten = 1.0;
				else TAtten = 0.0;
			}

			//specular

			half3 h = normalize(lightDir + viewDir);
			float nh = (dot(s.Normal, h));
			float spec = pow(nh, s.Gloss) * s.Specular;
			if (spec > .5) spec = 1.0;
			else spec = 0.0;
			half4 c;

			half3 ambient = _LightColor0.rgb * .3;
			half3 diffuse = _LightColor0.rgb * pow(NdotL * 0.5 + 0.5, 2.0);
			half3 specular = _LightColor0.rgb * spec;

			c.rgb = ((ambient + diffuse) * (TAtten * 2) * s.Albedo) + specular * (TAtten * 2);
			c.a = s.Alpha;
			return c;
		}

		struct Input {
			float4 grabUV;
			float2 uv_MainTex;
			float2 uv_NoiseMap;
			float2 uv_FlowMap;
		};


		sampler2D _MainTex;
		sampler2D _SecondaryTex;
		sampler2D _FlowMap;
		sampler2D _BackgroundTexture;
		sampler2D _NoiseMap;
		float4 _Color;

		float _Glossiness;
		float _Metallic;
		float _FlowCycleLength;
		float _FlowSpeed;

		float _WaveHeight;
		float _WaveSpeed;
		float _XFreq;
		float _ZFreq;

		void vert(inout appdata_full v, out Input o) {

			v.vertex.y += sin((_Time * _WaveSpeed) + (v.vertex.x * _XFreq)) * _WaveHeight;
			v.vertex.y += sin((_Time * _WaveSpeed) + (v.vertex.z * _ZFreq)) * _WaveHeight;

			v.vertex.y = v.vertex.y + _WaveHeight*2; // deferred rendering fix

			float4 hpos = UnityObjectToClipPos(v.vertex);
			o.grabUV = ComputeGrabScreenPos(hpos);
			o.uv_MainTex = v.texcoord;
			o.uv_FlowMap = v.texcoord1;
			o.uv_NoiseMap = v.texcoord2;
		}


		void surf(Input IN, inout SurfaceOutput o) {

			float2 flowMap = tex2D(_FlowMap, IN.uv_FlowMap).rg;
			float offset = tex2D(_NoiseMap, IN.uv_NoiseMap).r;

			flowMap = (flowMap * 2.0) - 1.0f;

			flowMap.r *= -1.0;

			// TIME BS

			float halfCycle = _FlowCycleLength * .5f;

			float time = _Time * _FlowSpeed;

			float timeOffset1 = fmod(time, _FlowCycleLength);
			float timeOffset2 = fmod(time + halfCycle, _FlowCycleLength);

			float p1 = (offset * 0.5) + timeOffset1;
			float p2 = (offset * 0.5) + timeOffset2;

			float3 norm1 = UnpackNormal(tex2D(_MainTex, IN.uv_MainTex + (flowMap * p1)));
			float3 norm2 = UnpackNormal(tex2D(_SecondaryTex, IN.uv_MainTex + (flowMap * p2)));

			float flow = abs(halfCycle - timeOffset1) / halfCycle;

			float3 normFinal = lerp(norm1, norm2, flow);

			//
			_Color.g *= offset;
			o.Albedo = tex2Dproj(_BackgroundTexture, UNITY_PROJ_COORD(IN.grabUV)) *_Color;
			o.Normal = normFinal;
			o.Specular = _Metallic;
			o.Gloss = _Glossiness;
		}
		ENDCG

		}
	FallBack "Diffuse"
}
