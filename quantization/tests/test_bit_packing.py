"""Bit layout and lossless round-trip checks; no model weights are needed."""

import unittest

import torch

from helper import pack_weights, unpack_weights


class BitPackingTests(unittest.TestCase):
    def test_two_bit_known_bytes(self):
        values = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], dtype=torch.uint8)
        packed = pack_weights(values, 2)
        self.assertEqual(packed.tolist(), [228, 27])
        self.assertEqual(packed.numel() * packed.element_size(), 2)
        self.assertTrue(torch.equal(unpack_weights(packed, 2), values))

    def test_four_bit_known_bytes(self):
        values = torch.tensor([0, 15, 1, 14], dtype=torch.uint8)
        packed = pack_weights(values, 4)
        self.assertEqual(packed.tolist(), [240, 225])
        self.assertTrue(torch.equal(unpack_weights(packed, 4), values))

    def test_one_bit_known_byte(self):
        values = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8)
        self.assertEqual(pack_weights(values, 1).tolist(), [170])
        self.assertEqual(unpack_weights(torch.tensor([170], dtype=torch.uint8), 1).tolist(), values.tolist())

    def test_eight_bit_is_identity(self):
        values = torch.tensor([0, 1, 127, 128, 255], dtype=torch.uint8)
        self.assertTrue(torch.equal(pack_weights(values, 8), values))
        self.assertTrue(torch.equal(unpack_weights(values, 8), values))

    def test_seeded_random_round_trips(self):
        generator = torch.Generator().manual_seed(42)
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                values = torch.randint(0, 1 << bits, (64,), dtype=torch.uint8, generator=generator)
                packed = pack_weights(values, bits)
                restored = unpack_weights(packed, bits)
                self.assertTrue(torch.equal(restored, values))
                self.assertEqual(packed.numel(), values.numel() * bits // 8)
                self.assertEqual(packed.dtype, torch.uint8)
                self.assertEqual(restored.dtype, torch.uint8)
                self.assertEqual(packed.device, values.device)
                self.assertEqual(restored.device, values.device)

    def test_zero_and_maximum_codes(self):
        for bits in (1, 2, 4, 8):
            for value, expected_byte in ((0, 0), ((1 << bits) - 1, 255)):
                with self.subTest(bits=bits, value=value):
                    values = torch.full((16,), value, dtype=torch.uint8)
                    packed = pack_weights(values, bits)
                    self.assertEqual(packed.tolist(), [expected_byte] * packed.numel())
                    self.assertTrue(torch.equal(unpack_weights(packed, bits), values))

    def test_every_possible_byte_can_be_repacked(self):
        packed = torch.arange(256, dtype=torch.uint8)
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                decoded = unpack_weights(packed, bits)
                self.assertLessEqual(decoded.max().item(), (1 << bits) - 1)
                self.assertTrue(torch.equal(pack_weights(decoded, bits), packed))

    def test_empty_tensor(self):
        empty = torch.empty(0, dtype=torch.uint8)
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                self.assertEqual(pack_weights(empty, bits).numel(), 0)
                self.assertEqual(unpack_weights(empty, bits).numel(), 0)

    def test_non_contiguous_inputs(self):
        values = torch.tensor([0, 9, 1, 9, 2, 9, 3, 9], dtype=torch.uint8)[::2]
        self.assertFalse(values.is_contiguous())
        self.assertEqual(pack_weights(values, 2).tolist(), [228])
        packed = torch.tensor([228, 99, 27, 99], dtype=torch.uint8)[::2]
        self.assertEqual(unpack_weights(packed, 2).tolist(), [0, 1, 2, 3, 3, 2, 1, 0])

    def test_rejects_unsupported_bits(self):
        values = torch.zeros(8, dtype=torch.uint8)
        for bits in (0, -1, 3, 5, 16, True, 2.0, "2", None):
            for function in (pack_weights, unpack_weights):
                with self.subTest(bits=bits, function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(values, bits)

    def test_rejects_wrong_dtype_or_non_tensor(self):
        for values in (torch.zeros(8), torch.zeros(8, dtype=torch.int8), [0] * 8):
            for function in (pack_weights, unpack_weights):
                with self.subTest(function=function.__name__):
                    with self.assertRaises(TypeError):
                        function(values, 2)

    def test_rejects_non_vector_inputs(self):
        for values in (torch.tensor(0, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.uint8)):
            for function in (pack_weights, unpack_weights):
                with self.subTest(function=function.__name__, shape=values.shape):
                    with self.assertRaises(ValueError):
                        function(values, 2)

    def test_rejects_unaligned_length(self):
        for bits, length in ((1, 7), (2, 3), (4, 3)):
            with self.subTest(bits=bits):
                with self.assertRaises(ValueError):
                    pack_weights(torch.zeros(length, dtype=torch.uint8), bits)

    def test_rejects_out_of_range_codes(self):
        for bits in (1, 2, 4):
            with self.subTest(bits=bits):
                values = torch.zeros(8 // bits, dtype=torch.uint8)
                values[0] = 1 << bits
                with self.assertRaises(ValueError):
                    pack_weights(values, bits)


if __name__ == "__main__":
    unittest.main()
