using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LoadingAnimation : MonoBehaviour
{
    private RectTransform icon_rect;
    public float time_step;
    public float angle_step;
    private float start_time;
    // Start is called before the first frame update
    void Start()
    {
        icon_rect = GetComponent<RectTransform>();
        start_time = 0f;
    }

    // Update is called once per frame
    void Update()
    {
        start_time += Time.deltaTime;

        if(start_time >= time_step)
        {
            Vector3 ang = icon_rect.localEulerAngles;
            ang.z -= angle_step;
            icon_rect.localEulerAngles = ang;
            start_time = 0f;
        }
    }
}
