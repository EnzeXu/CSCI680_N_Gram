public class FunctionsTest extends CascadingTestCase { @ Test public void testPartialCounts ( ) { Function function = new AggregateBy . CompositeFunction ( new Fields ( "value" ) , Fields . ALL , new CountBy . CountPartials ( new Fields ( "count" ) ) , 2 ) ; Fields incoming = new Fields ( "value" ) ; TupleEntry [ ] tuples = new TupleEntry [ ] { new TupleEntry ( incoming , new Tuple ( "a" ) ) , new TupleEntry ( incoming , new Tuple ( "a" ) ) , new TupleEntry ( incoming , new Tuple ( "b" ) ) , new TupleEntry ( incoming , new Tuple ( "b" ) ) , new TupleEntry ( incoming , new Tuple ( "c" ) ) , new TupleEntry ( incoming , new Tuple ( "c" ) ) , new TupleEntry ( incoming , new Tuple ( "a" ) ) , new TupleEntry ( incoming , new Tuple ( "a" ) ) , new TupleEntry ( incoming , new Tuple ( "d" ) ) , new TupleEntry ( incoming , new Tuple ( "d" ) ) , } ; List < Tuple > expected = new ArrayList < Tuple > ( ) ; expected . add ( new Tuple ( "a" , 2L ) ) ; expected . add ( new Tuple ( "b" , 2L ) ) ; expected . add ( new Tuple ( "c" , 2L ) ) ; expected . add ( new Tuple ( "a" , 2L ) ) ; expected . add ( new Tuple ( "d" , 2L ) ) ; TupleListCollector collector = invokeFunction ( function , tuples , new Fields ( "value" , "count" ) ) ; Iterator < Tuple > iterator = collector . iterator ( ) ; int count = 0 ; while ( iterator . hasNext ( ) ) { count++ ; Tuple result = iterator . next ( ) ; int index = expected . indexOf ( result ) ; assertTrue ( index > -1 ) ; assertEquals ( result , expected . get ( index ) ) ; expected . remove ( index ) ; } assertEquals ( 5 , count ) ; } @ Test public void testPartialSums ( ) { Function function = new AggregateBy . CompositeFunction ( new Fields ( "key" ) , new Fields ( "value" ) , new SumBy . SumPartials ( new Fields ( "sum" ) , float . class ) , 2 ) ; Fields incoming = new Fields ( "key" , "value" ) ; TupleEntry [ ] tuples = new TupleEntry [ ] { new TupleEntry ( incoming , new Tuple ( "a" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "a" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "b" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "b" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "c" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "c" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "a" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "a" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "d" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "d" , 1 ) ) , } ; List < Tuple > expected = new ArrayList < Tuple > ( ) ; expected . add ( new Tuple ( "a" , 2F ) ) ; expected . add ( new Tuple ( "b" , 2F ) ) ; expected . add ( new Tuple ( "c" , 2F ) ) ; expected . add ( new Tuple ( "a" , 2F ) ) ; expected . add ( new Tuple ( "d" , 2F ) ) ; TupleListCollector collector = invokeFunction ( function , tuples , new Fields ( "key" , "sum" ) ) ; Iterator < Tuple > iterator = collector . iterator ( ) ; int count = 0 ; while ( iterator . hasNext ( ) ) { count++ ; Tuple result = iterator . next ( ) ; int index = expected . indexOf ( result ) ; assertTrue ( index > -1 ) ; assertEquals ( result , expected . get ( index ) ) ; expected . remove ( index ) ; } assertEquals ( 5 , count ) ; } @ Test public void testLocallyPartialSums ( ) { Function function = new AggregateByLocally . CompositeFunction ( new Fields ( "key" ) , new Fields ( "value" ) , new SumByLocally . SumPartials ( new Fields ( "sum" ) , float . class ) , 2 ) ; Fields incoming = new Fields ( "key" , "value" ) ; TupleEntry [ ] tuples = new TupleEntry [ ] { new TupleEntry ( incoming , new Tuple ( "a" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "a" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "b" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "b" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "c" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "c" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "a" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "a" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "d" , 1 ) ) , new TupleEntry ( incoming , new Tuple ( "d" , 1 ) ) , } ; List < Tuple > expected = new ArrayList < Tuple > ( ) ; expected . add ( new Tuple ( "a" , 2F ) ) ; expected . add ( new Tuple ( "b" , 2F ) ) ; expected . add ( new Tuple ( "c" , 2F ) ) ; expected . add ( new Tuple ( "a" , 2F ) ) ; expected . add ( new Tuple ( "d" , 2F ) ) ; TupleListCollector collector = invokeFunction ( function , tuples , new Fields ( "key" , "sum" ) ) ; Iterator < Tuple > iterator = collector . iterator ( ) ; int count = 0 ; while ( iterator . hasNext ( ) ) { count++ ; Tuple result = iterator . next ( ) ; int index = expected . indexOf ( result ) ; assertTrue ( index > -1 ) ; assertEquals ( result , expected . get ( index ) ) ; expected . remove ( index ) ; } assertEquals ( 5 , count ) ; } @ Test public void testLocallyPartialCounts ( ) { Function function = new AggregateByLocally . CompositeFunction ( new Fields ( "value" ) , Fields . ALL , new CountByLocally . CountPartials ( new Fields ( "count" ) ) , 2 ) ; Fields incoming = new Fields ( "value" ) ; TupleEntry [ ] tuples = new TupleEntry [ ] { new TupleEntry ( incoming , new Tuple ( "a" ) ) , new TupleEntry ( incoming , new Tuple ( "a" ) ) , new TupleEntry ( incoming , new Tuple ( "b" ) ) , new TupleEntry ( incoming , new Tuple ( "b" ) ) , new TupleEntry ( incoming , new Tuple ( "c" ) ) , new TupleEntry ( incoming , new Tuple ( "c" ) ) , new TupleEntry ( incoming , new Tuple ( "a" ) ) , new TupleEntry ( incoming , new Tuple ( "a" ) ) , new TupleEntry ( incoming , new Tuple ( "d" ) ) , new TupleEntry ( incoming , new Tuple ( "d" ) ) , } ; List < Tuple > expected = new ArrayList < Tuple > ( ) ; expected . add ( new Tuple ( "a" , 2L ) ) ; expected . add ( new Tuple ( "b" , 2L ) ) ; expected . add ( new Tuple ( "c" , 2L ) ) ; expected . add ( new Tuple ( "a" , 2L ) ) ; expected . add ( new Tuple ( "d" , 2L ) ) ; TupleListCollector collector = invokeFunction ( function , tuples , new Fields ( "value" , "count" ) ) ; Iterator < Tuple > iterator = collector . iterator ( ) ; int count = 0 ; while ( iterator . hasNext ( ) ) { count++ ; Tuple result = iterator . next ( ) ; int index = expected . indexOf ( result ) ; assertTrue ( index > -1 ) ; assertEquals ( result , expected . get ( index ) ) ; expected . remove ( index ) ; } assertEquals ( 5 , count ) ; } }