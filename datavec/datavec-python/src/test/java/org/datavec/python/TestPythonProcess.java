package org.datavec.python;

import org.junit.Assert;
import org.junit.Test;

public class TestPythonProcess {

    @Test
    public void testPythonProcess() throws Exception{
        String stdout = PythonProcess.runAndReturn("-m", "pip", "list");
        System.out.println(stdout);
        Assert.assertTrue(stdout.replace(" ", "").contains("PackageVersion"));
    }
    @Test
    public void testPackageVersion() throws Exception{
        System.out.println(PythonProcess.getPackageVersion("numpy"));
    }

    @Test
    public void testPackageInstalledCheck() throws Exception{
        Assert.assertFalse(PythonProcess.isPackageInstalled("abcdefgh"));
    }

}
