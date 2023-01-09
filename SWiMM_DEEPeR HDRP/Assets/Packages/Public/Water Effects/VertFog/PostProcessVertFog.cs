using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[ExecuteInEditMode]
public class PostProcessVertFog : MonoBehaviour {


    [Range(0.00001f, 1.0f)] public float scale = 0.0f;
    public float height = 0.0f;

    [Header("Wave")]
    public float waveHeight = 0.0f;
    public float waveSpeed = 0.0f;
    [Range(-1,1)]public float waveXFreq = 0.0f;
    [Range(-1,1)]public float waveZFreq = 0.0f;


    public Color color;
    private Material mat;

    void Start()
    {
        mat = new Material(Shader.Find("Hidden/PostProcessVertFog"));
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        //UnityEditorInternal.ComponentUtility.MoveComponentDown(this);
        var p = GL.GetGPUProjectionMatrix(GetComponent<Camera>().projectionMatrix, false);
        p[2, 3] = p[3, 2] = 0.0f;
        p[3, 3] = 1.0f;

        var clipToWorld = Matrix4x4.Inverse(p * GetComponent<Camera>().worldToCameraMatrix) * Matrix4x4.TRS(new Vector3(0, 0, -p[2, 2]), Quaternion.identity, Vector3.one);
        mat.SetMatrix("_ClipToWorld", clipToWorld);

        mat.SetFloat("_Threshold", scale);
        mat.SetFloat("_Depth", height);
        mat.SetColor("_Color", color);

        mat.SetFloat("_WaveHeight", waveHeight);
        mat.SetFloat("_WaveSpeed", waveSpeed);
        mat.SetFloat("_XFrequency", waveXFreq);
        mat.SetFloat("_ZFrequency", waveZFreq);

        Graphics.Blit(source, destination, mat);
    }
}
