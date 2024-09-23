public class OffsetRecorderIterator<K, V> implements Iterator<ConsumerRecord<K, V>> { private final Map<TopicPartition, OffsetAndMetadata> currentOffsets; private final Iterator<ConsumerRecord<K, V>> iterator; private String topic; private int partition; private long offset; public OffsetRecorderIterator( Map<TopicPartition, OffsetAndMetadata> currentOffsets, Iterator<ConsumerRecord<K, V>> iterator ) { this.currentOffsets = currentOffsets; this.iterator = iterator; } @Override public boolean hasNext() { boolean hasNext = iterator.hasNext(); if( !hasNext ) addLastOffset(); return hasNext; } @Override public ConsumerRecord<K, V> next() { addLastOffset(); ConsumerRecord<K, V> next = iterator.next(); topic = next.topic(); partition = next.partition(); offset = next.offset(); return next; } private void addLastOffset() { if( topic == null ) return; currentOffsets.put( new TopicPartition( topic, partition ), new OffsetAndMetadata( offset + 1 ) ); } }