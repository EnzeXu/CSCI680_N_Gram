public class Blort { public static int maybeThrow(int x) { if (x < 10) { throw new RuntimeException(); } return x; } public static int exTest1(int x) { try { maybeThrow(x); return 1; } catch (RuntimeException ex) { return 2; } } public static int exTest2(int x) { try { x++; x = maybeThrow(x); } catch (RuntimeException ex) { return 1; } try { x++; } catch (RuntimeException ex) { return 2; } try { return maybeThrow(x); } catch (RuntimeException ex) { return 3; } } }