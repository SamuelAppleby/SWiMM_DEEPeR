using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Floater : MonoBehaviour
{
    public float buoyant_strength;
    private Rigidbody rb;
    private SphereCollider m_collider;
    public Collider water_collider;      // Can be overriden onCollision, but default to avoid skipping physics ticks
    public LayerMask water_mask;
    private float distance_underwater;

    void Start()
    {
        rb = GetComponentInParent<Rigidbody>();
        m_collider = GetComponent<SphereCollider>();
        m_collider.radius = .15f;
        m_collider.isTrigger = true;
    }

    private void Update()
    {
        distance_underwater = water_collider.transform.position.y - transform.position.y;
    }

    private void FixedUpdate()
    {
        if (distance_underwater > 0f)
        {
            float top_of_water = water_collider.transform.position.y + water_collider.bounds.extents.y;
            float bottom_of_collider = GetComponent<Collider>().transform.position.y - m_collider.radius;
            float diff = Math.Clamp(top_of_water - bottom_of_collider, 0, (2 * m_collider.radius));
            float floater_ratio_underwater = diff / (2 * m_collider.radius);
            rb.AddForceAtPosition(Vector3.up * floater_ratio_underwater * buoyant_strength, transform.position, ForceMode.Force);
        }
    }
}
