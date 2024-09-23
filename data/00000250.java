public class RPoint { public float x; public float y; public RPoint(float x,float y) { this.x = x; this.y = y; } public RPoint(double x, double y) { this.x = (float)x; this.y = (float)y; } public RPoint() { x = 0; y = 0; } public RPoint(RPoint p) { this.x = p.x; this.y = p.y; } float getX() { return this.x; } float getY() { return this.y; } void setLocation(float nx, float ny) { this.x = nx; this.y = ny; } public void transform(RMatrix m) { float tempx = m.m00*x + m.m01*y + m.m02; float tempy = m.m10*x + m.m11*y + m.m12; x = tempx; y = tempy; } public void translate(float tx, float ty) { x += tx; y += ty; } public void translate(RPoint t) { x += t.x; y += t.y; } public void rotate(float angle, float vx, float vy) { float c = (float)Math.cos(angle); float s = (float)Math.sin(angle); x -= vx; y -= vy; float tempx = x; float tempy = y; x = tempx*c - tempy*s; y = tempx*s + tempy*c; x += vx; y += vy; } public void rotate(float angle) { float c = (float)Math.cos(angle); float s = (float)Math.sin(angle); float tempx = x; float tempy = y; x = tempx*c - tempy*s; y = tempx*s + tempy*c; } public void rotate(float angle, RPoint v) { float c = (float)Math.cos(angle); float s = (float)Math.sin(angle); x -= v.x; y -= v.y; float tempx = x; float tempy = y; x = tempx*c - tempy*s; y = tempx*s + tempy*c; x += v.x; y += v.y; } public void scale (float sx, float sy) { x *= sx; y *= sy; } public void scale (float s) { x *= s; y *= s; } public void scale (RPoint s) { x *= s.x; y *= s.y; } public void normalize () { float norma = norm(); if(norma!=0) scale(1/norma); } public void sub (RPoint p) { x -= p.x; y -= p.y; } public void add (RPoint p) { x += p.x; y += p.y; } public float mult (RPoint p) { return (x * p.x + y * p.y); } public RPoint cross (RPoint p) { return new RPoint(x * p.y - p.x * y, y * p.x - p.y * x); } public float norm () { return (float)Math.sqrt(mult(this)); } public float sqrnorm () { return (float)mult(this); } public float angle (RPoint p) { float normp = p.norm(); float normthis = norm(); return (float)Math.acos(mult(p)/(normp*normthis)); } public float dist (RPoint p) { float dx = (p.x-this.x); float dy = (p.y-this.y); return (float)Math.sqrt(dx*dx + dy*dy); } public void print(){ System.out.print("("+x+","+y+")\n"); } }