public class Blort { private static RuntimeException theException = new RuntimeException(); public static void test1() { throw theException; } public static int test2() { try { throw theException; } catch (RuntimeException ex) { return 1; } } }