public class HashJoinAroundHashJoinLeftMostGraph extends FlowElementGraph { public HashJoinAroundHashJoinLeftMostGraph ( ) { Map sources = new HashMap ( ) ; sources . put ( "lower" , new NonTap ( "lower" , new Fields ( "offset" , "line" ) ) ) ; sources . put ( "upper1" , new NonTap ( "upper" , new Fields ( "offset" , "line" ) ) ) ; sources . put ( "upper2" , new NonTap ( "upper" , new Fields ( "offset" , "line" ) ) ) ; Map sinks = new HashMap ( ) ; sinks . put ( "sink" , new NonTap ( "sink" , new Fields ( "offset" , "line" ) ) ) ; Function splitter = new RegexSplitter ( new Fields ( "num" , "char" ) , " " ) ; Pipe pipeLower = new Each ( new Pipe ( "lower" ) , new Fields ( "line" ) , splitter ) ; Pipe pipeUpper1 = new Each ( new Pipe ( "upper1" ) , new Fields ( "line" ) , splitter ) ; Pipe pipeUpper2 = new Each ( new Pipe ( "upper2" ) , new Fields ( "line" ) , splitter ) ; Pipe splice1 = new HashJoin ( pipeUpper1 , new Fields ( "num" ) , pipeUpper2 , new Fields ( "num" ) , new Fields ( "num1" , "char1" , "num2" , "char2" ) ) ; splice1 = new Each ( splice1 , new Identity ( ) ) ; Pipe splice2 = new HashJoin ( "sink" , splice1 , new Fields ( "num1" ) , pipeLower , new Fields ( "num" ) , new Fields ( "num1" , "char1" , "num2" , "char2" , "num3" , "char3" ) ) ; initialize ( sources , sinks , splice2 ) ; } }