public class ScalaCompilerLoader extends URLClassLoader { private final ClassLoader sbtLoader; public ScalaCompilerLoader(URL[] urls, ClassLoader sbtLoader) { super(urls, ClasspathUtil.rootLoader()); this.sbtLoader = sbtLoader; } @Override public Class<?> loadClass(String className, boolean resolve) throws ClassNotFoundException { if (className.startsWith("xsbti.")) { Class<?> c = sbtLoader.loadClass(className); if (resolve) resolveClass(c); return c; } else { return super.loadClass(className, resolve); } } }