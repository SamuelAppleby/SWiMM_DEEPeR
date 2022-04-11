using System;
using UnityEngine;
using UnityStandardAssets.Utility;

namespace UnityStandardAssets.Characters.FirstPerson
{
    public class HeadBob : MonoBehaviour
    {
        public Camera Camera;
        public CurveControlledBob motionBob = new CurveControlledBob();
        public LerpControlledBob jumpAndLandingBob = new LerpControlledBob();
        public ThirdPersonMovement rigidbodyFirstPersonController;
        public float StrideInterval;
        [Range(0f, 1f)] public float RunningStrideLengthen;
        private Vector3 m_OriginalCameraPosition;

        private void Start()
        {
            motionBob.Setup(Camera, StrideInterval);
            m_OriginalCameraPosition = Camera.transform.localPosition;
        }

        private void Update()
        {
            Vector3 newCameraPosition = Camera.transform.localPosition;
            newCameraPosition.y = m_OriginalCameraPosition.y - jumpAndLandingBob.Offset();
            Camera.transform.localPosition = newCameraPosition;
        }
    }
}
