public class Blort { public synchronized void testInstance1() { } public synchronized void testInstance2(Object x) { x.hashCode(); } public synchronized int testInstance3(int x, int y, int z) { if (x == 1) { return 1; } else { return 2; } } public synchronized long testInstance4(long x) { if (x == 1) { return 1; } else { return 2; } } public synchronized void testInstance5() { testInstance5(); } public static synchronized void testStatic1() { } public static synchronized void testStatic2(Object x) { x.hashCode(); } public static synchronized int testStatic3(int x, int y, int z) { if (x == 1) { return 1; } else { return 2; } } public static synchronized long testStatic4(long x) { if (x == 1) { return 1; } else { return 2; } } public static synchronized void testStatic5() { testStatic5(); } }