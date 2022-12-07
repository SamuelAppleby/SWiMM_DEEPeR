using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class PostProcessingWater : MonoBehaviour {

    [HideInInspector] public Material blurMaterial;
    [HideInInspector] public Material distortionMaterial;

    [Range(0, 3)] public float blurResolution = 1;

    public Color color;

    [Header("Distortion")]
    public Texture distortionMap;
    public float xScale = 1.0f;
    public float yScale = 1.0f;
    [Range(0,1)]public float strength = 1.0f;

    public float xSpeed;
    public float ySpeed;


    private Material blurMat;
    private Material distortionMat;

    private RenderTexture blurTex;
    private RenderTexture distortionTex;

    private bool awake;

    // Use this for initialization
    void Awake()
    {
        distortionMat = new Material(Shader.Find("Nasty-Water/Distortion"));
        blurMat = new Material(Shader.Find("Nasty-Water/GBlur"));
        awake = true;
    }

    // Update is called once per frame
    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (awake)
        {
            distortionTex = RenderTexture.GetTemporary(source.width, source.height);
            awake = false;
        }

        if (blurMaterial != null && distortionMaterial != null)
        {
            blurMat = blurMaterial;
            distortionMat = distortionMaterial;
        }
        else
        {
            blurMat.SetFloat("_ResX", Screen.width * blurResolution);
            blurMat.SetFloat("_ResY", Screen.height * blurResolution);

            distortionMat.SetTexture("_Normal", distortionMap);

            distortionMat.SetFloat("_XScale", xScale);
            distortionMat.SetFloat("_YScale", yScale);
            distortionMat.SetFloat("_Strength", strength);

            distortionMat.SetFloat("_XSpeed", xSpeed);
            distortionMat.SetFloat("_YSpeed", ySpeed);

            distortionMat.SetColor("_Color", color);
        }

        Graphics.Blit(source, distortionTex, distortionMat);
        Graphics.Blit(distortionTex, destination, blurMat);
    }

}
