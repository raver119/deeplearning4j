package org.nd4j;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tensorflow.TFOpImportFilter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.protobuf.ByteString;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.*;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;

import static org.junit.Assert.assertEquals;

public class TestSavedModel {

    @Test
    public void testSavedModelFull() throws Exception {

//        SavedModelBundle.Loader l = SavedModelBundle.loader("C:\\Temp\\TF_SavedModel\\testModel");

        //First: work out tags
        String path = "C:\\Temp\\TF_SavedModel\\testModel";
        String pathPb = path + "\\saved_model.pb";
        byte[] bytes = FileUtils.readFileToByteArray(new File(pathPb));
        List<MetaGraphDef> defs = SavedModel.parseFrom(bytes).getMetaGraphsList();
        for(MetaGraphDef mgd : defs){
            List<ByteString> bstrings = mgd.getMetaInfoDef().getTagsList().asByteStringList();
            for (ByteString bs : bstrings){
                System.out.println("TAG: " + bs.toStringUtf8());
            }
        }

        SavedModelBundle smb = SavedModelBundle.load("C:\\Temp\\TF_SavedModel\\testModel", "serve");

        Graph g = smb.graph();

        Iterator<Operation> op = g.operations();
        while(op.hasNext()){
            Operation o = op.next();
            System.out.println(o);
        }

        byte[] graphDefBytes = g.toGraphDef();
        GraphDef gd = GraphDef.parseFrom(graphDefBytes);
        int nc = gd.getNodeCount();
        System.out.println(nc);

        for( int i=0; i<nc; i++ ){
            System.out.println("GRAPH DEF OP: " + i + " - " + gd.getNode(i).getOp() + " - " + gd.getNode(i).getName());
        }

        SameDiff sd = SameDiff.importFrozenTF(gd);

        INDArray in = Nd4j.createFromArray(0.65641118f, 0.80981255f, 0.87217591f, 0.9646476f).reshape(1, 4);
        INDArray expOut = Nd4j.createFromArray(0.3246807f, 0.26819962f, 0.40711975f).reshape(1, 3);

        INDArray out = sd.outputSingle(Collections.singletonMap("input", in), "softmax");
        assertEquals(expOut, out);
    }


    @Test
    public void testSavedModel() throws Exception {
        SavedModelBundle smb = SavedModelBundle.load("C:\\Temp\\TF_SavedModel\\testModel", "serve");
        Graph g = smb.graph();
        byte[] graphDefBytes = g.toGraphDef();
        GraphDef gd = GraphDef.parseFrom(graphDefBytes);

        List<Tensor<?>> tfOut = smb.session().runner()
                .fetch("Variable")
                .fetch("Variable_1")
                .run();
        System.out.println(tfOut);
        INDArray w = convert(tfOut.get(0));
        INDArray b = convert(tfOut.get(1));

        final Set<String> ignore = new HashSet<>(Arrays.asList("SaveV2", "Assign", "StringJoin", "ShardedFilename", "Pack", "MergeV2Checkpoints", "RestoreV2"));
        SameDiff sd = TFGraphMapper.importGraph(gd, null, new TFOpImportFilter() {
            @Override
            public boolean skipOp(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
                return ignore.contains(nodeDef.getOp()) || nodeDef.getName().startsWith("save/");
            }
        });
        sd.getVariable("Variable").getArr().assign(w);
        sd.getVariable("Variable_1").getArr().assign(b);

        INDArray in = Nd4j.createFromArray(0.65641118f, 0.80981255f, 0.87217591f, 0.9646476f).reshape(1, 4);
        INDArray expOut = Nd4j.createFromArray(0.3246807f, 0.26819962f, 0.40711975f).reshape(1, 3);

        System.out.println(sd.summary());

        INDArray out = sd.outputSingle(Collections.singletonMap("input", in), "Softmax");
        assertEquals(expOut, out);
    }

    private static INDArray convert(Tensor<?> t){
        FloatBuffer fb = FloatBuffer.allocate(t.numElements());
        t.writeTo(fb);
        float[] fArr = new float[fb.position()];
        for( int i=0; i<fArr.length; i++ ){
            fArr[i] = fb.get(i);
        }
        long[] shape = t.shape();
        return Nd4j.createFromArray(fArr).reshape(shape);
    }
}
