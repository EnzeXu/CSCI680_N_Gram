public class KeyguardManagerCompat { private static Method isKeyguardSecure; private static boolean available; static { try { isKeyguardSecure = KeyguardManager.class.getMethod("isKeyguardSecure", (Class[]) null); available = true; } catch (Exception e) { available = false; } } public static boolean isKeyguardSecure(KeyguardManager inst) { if (!available) { return false; } try { return (boolean) isKeyguardSecure.invoke(inst, null); } catch (Exception e) { return false; } } }