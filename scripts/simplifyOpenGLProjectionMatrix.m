% This snippet shows that the MonoGS getProjectionMatrix2 function can be simplified greatly.
syms cx cy fx fy W H zfar znear z_sign

left = ((2 * cx - W) / W - 1.0) * W / 2.0;
right = ((2 * cx - W) / W + 1.0) * W / 2.0;
top = ((2 * cy - H) / H + 1.0) * H / 2.0;
bottom = ((2 * cy - H) / H - 1.0) * H / 2.0;
left = znear / fx * left;
right = znear / fx * right;
top = znear / fy * top;
bottom = znear / fy * bottom;

P = sym(zeros(4, 4));
P(1, 1) = 2.0 * znear / (right - left);
P(2, 2) = 2.0 * znear / (top - bottom);
P(1, 3) = (right + left) / (right - left);
P(2, 3) = (top + bottom) / (top - bottom);
P(4, 3) = 1.0;
P(3, 3) = 1.0 * zfar / (zfar - znear);
P(3, 4) = -(zfar * znear) / (zfar - znear);

simplify(P)