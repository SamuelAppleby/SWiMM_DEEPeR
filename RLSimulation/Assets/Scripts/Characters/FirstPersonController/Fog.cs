using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fog : MonoBehaviour
{
    public ThirdPersonMovement player;

    public float fogStart = 100;
    public float fogEnd = 200;

    static public void SetFogStart(float start)
    {
        RenderSettings.fogStartDistance = start;
    }

    void Start()
    {
        RenderSettings.fogStartDistance = fogStart;
        RenderSettings.fogEndDistance = fogEnd;
    }

    void Update()
    {
    }

    private void OnPostRender()
    {
        RenderSettings.fog = player.m_IsUnderwater;
    }
}
