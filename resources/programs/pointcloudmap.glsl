#version 330

#if defined VERTEX_SHADER

// Vertex attributes
in vec3 in_vert;    // Vertex position
in vec3 in_color;   // Vertex color

// Uniform matrices for transformations
uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

// Pass the color to the fragment shader
out vec3 v_color;

void main() {
    // Compute the transformed position
    gl_Position = m_proj * m_camera * m_model * vec4(in_vert, 1.0);
    // Pass the vertex color through
    v_color = in_color;
}

#elif defined FRAGMENT_SHADER

// Receive the color from the vertex shader
in vec3 v_color;
// Output fragment color
out vec4 f_color;

void main() {
    // Simply output the color with full opacity
    f_color = vec4(v_color, 1.0);
}

#endif
