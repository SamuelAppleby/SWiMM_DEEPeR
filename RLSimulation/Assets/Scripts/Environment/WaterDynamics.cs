using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaterDynamics : MonoBehaviour
{
    private const float water_density = 1025;
    private const float g = 9.81f;
    private BoxCollider m_collider;
    private Vector3 top_of_collider_pos;

    // Start is called before the first frame update
    void Start()
    {
        m_collider = GetComponent<BoxCollider>();
        top_of_collider_pos = transform.position + m_collider.center + new Vector3(0, m_collider.size.y / 2, 0);
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnCollisionStay(Collision other)
    {
        ApplyBuoyantForces(other.collider);
    }

    private void OnTriggerStay(Collider other)
    {
        ApplyBuoyantForces(other);
    }


    private void ApplyBuoyantForces(Collider other)
    {
        /* Buouancy */
        BoxCollider other_box = other as BoxCollider;
        Rigidbody other_rb = other.GetComponentInParent<Rigidbody>();

        if (other_box != null && other_rb != null)
        {
            Vector3 bottom_of_collider_pos = other_box.transform.position + other_box.center - new Vector3(0, other_box.size.y / 2, 0);

            float distance_underwater = top_of_collider_pos.y - bottom_of_collider_pos.y;
            float ratio_underwater = Math.Clamp(distance_underwater / other_box.size.y, 0, 1);
            float volume_displaced = ratio_underwater * (other_box.size.x * other_box.size.y * other_box.size.z);
            float buoyant_force = water_density * g * volume_displaced;       // Fb = -pgV

            other_rb.AddForce(new Vector3(0, buoyant_force, 0), ForceMode.Force);
        }
    }
}
