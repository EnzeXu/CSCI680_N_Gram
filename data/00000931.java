public class DoubleUtil { public static double remainderWithFix ( double value , int remainder ) { double res = value % remainder ; return res < 0 ? res += remainder : res ; } }