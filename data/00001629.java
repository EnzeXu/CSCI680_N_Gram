public class GroupingPartitioner extends HasherPartitioner implements Partitioner < Tuple , Tuple > { public int getPartition ( Tuple key , Tuple value , int numReduceTasks ) { return ( hashCode ( key ) & Integer . MAX_VALUE ) % numReduceTasks ; } @ Override public void configure ( JobConf job ) { setConf ( job ) ; } }