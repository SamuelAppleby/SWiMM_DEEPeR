using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaterVolume : MonoBehaviour {

    bool isVisable = false;

    public Material blurMaterial;
    public Material distortionMaterial;

    //
    private PostProcessingWater PostEffect_water;
    //

    void Update()
    {
        isVisable = GetComponent<BoxCollider>().bounds.Contains(Camera.main.transform.position);

        if (isVisable)
        {

            if (PostEffect_water == null)
            {
                PostEffect_water = Camera.main.gameObject.AddComponent<PostProcessingWater>();
                PostEffect_water.blurMaterial = blurMaterial;
                PostEffect_water.distortionMaterial = distortionMaterial;
            }
            else
            { 
                PostEffect_water.enabled = true;
            }
        }
        else
        {
            if (PostEffect_water != null)
                PostEffect_water.enabled = false;
        }
    }
}
