public class WhenNoParametersMatch { @ DataPoints public static int [ ] ints = { 0 , 1 , 3 , 5 , 1776 } ; @ DataPoints public static Matcher < ? > [ ] matchers = { not ( 0 ) , is ( 1 ) } ; @ RunWith ( Theories . class ) public static class AssumptionsFail { @ DataPoint public static int DATA = 0 ; @ DataPoint public static Matcher < Integer > MATCHER = null ; @ Theory public void nonZeroIntsAreFun ( int x ) { assumeThat ( x , MATCHER ) ; } } @ Theory public void showFailedAssumptionsWhenNoParametersFound ( int data , Matcher < Integer > matcher ) throws Exception { assumeThat ( data , not ( matcher ) ) ; AssumptionsFail . DATA = data ; AssumptionsFail . MATCHER = matcher ; String result = testResult ( AssumptionsFail . class ) . toString ( ) ; assertThat ( result , containsString ( matcher . toString ( ) ) ) ; assertThat ( result , containsString ( "" + data ) ) ; } }