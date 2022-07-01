using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaterDynamics : MonoBehaviour
{
    public const float water_density = 1025;
    private BoxCollider m_collider;
    private Vector3 top_of_collider_pos;

    void Start()
    {
        m_collider = GetComponent<BoxCollider>();
        top_of_collider_pos = transform.position + m_collider.center + new Vector3(0, m_collider.size.y / 2, 0);
    }

    //private void OnCollisionStay(Collision other)
    //{
    //    Rigidbody other_rb = other.gameObject.GetComponentInParent<Rigidbody>();

    //    if (other.gameObject.tag.Equals("Floater") && other_rb)
    //    {
    //        ApplyBuoyantForces(other.transform.position, other_rb);
    //    }
    //}

    //private void OnTriggerStay(Collider other)
    //{
    //    Rigidbody other_rb = other.gameObject.GetComponentInParent<Rigidbody>();

    //    if (other.gameObject.tag.Equals("Floater") && other_rb)
    //    {
    //        ApplyBuoyantForces(other.transform.position, other_rb);
    //    }
    //}

    //private void ApplyBuoyantForces(Vector3 position, Rigidbody parent_rb)
    //{
    //    float distance_underwater = top_of_collider_pos.y - position.y;
    //    distance_underwater = Math.Clamp(distance_underwater, 0.0f, 4.0f);
    //    parent_rb.AddForceAtPosition(Vector3.up * Math.Abs(distance_underwater) * 50, position, ForceMode.Force);
    //}
}
