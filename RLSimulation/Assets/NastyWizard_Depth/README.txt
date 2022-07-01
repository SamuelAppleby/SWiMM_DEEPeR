//////////////////////////////
NASTY OLD WIZARD FOG AND WATER
//////////////////////////////

Twitter: @casey_macneil
Email: detrocasey@gmail.com

//////////////////////////////

To use the water shader, simply make a material and set the shader to water. The texture it takes in is a normal map that is used for refraction, it's not required.
There are 3 values for depth: Color, Scale, and Height
Scale determins how faded the edge of the fog is, height determins where it starts and stops.

For the vertical fog, simply select your camera and add the script "PostProcessVertFog" to it, the values are the same as the water.

UPDATE:

There are now two separate shaders for water, the original toon looking water, and the new more realistic water. 
I have also added a tool for generating a water mesh of variable size, to use it, simply make an empy game object and add
the NastyWater.cs script to it. All you need to do is type in the width and the height and hit GENERATE MESH.
If you want the water to use a post effect on the camera when it enters the water, that takes a bit more work currently (working on simplifying this).
The fog post effect now allows you to use waves on it, just play with the values until your fog waves are moving the way you want.
All shaders for this asset are now under Nasty-Water.

Video tutorial for tools and shader use is coming soon.

Under Water Post effect:
	step 1 - Add the WaterVolume.cs Script to your water object, and make sure you have a BoxCollider on the object set to trigger.
	step 2 - Make a material using Nasty-Water/Distortion, and one using Nasty-Water/GBlur as the shader.
	step 3 - Add these materials to the WaterVolume component.
	step 4 - Play with the values on both materials until you get the desired effect.